package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unsafe"
)

const (
	IN                    = "glove.6B.100d.txt"
	OUT                   = "out.par"
	NR_DIM                = 100        // number of dimensions in embedding vector space
	NR_SMAMPLE_DIM        = 10         // number of dimensions form NR_DIM to take as a sample
	NR_VEC                = 1024       // number of vectors to store in a bucket
	NR_BUCKETS            = 1024       // number of buckets stored in a partition
	NR_PARTITIONS         = 512        // number of partitions
	NULL_VAL       uint32 = 4294967295 // (2^32)-1
)

type Vector struct {
	id     uint32
	vector []float64
	end    string
}

type PartitionTuple struct {
	dim   int
	score float64
}

type PartResult struct {
	buckIdx int
	score   float64
}

type Bucket struct {
	tuples []BucketTuple
}

type BucketTuple struct {
	id    uint32  // embedding index, unique id
	score float64 // embedding score
}

var VecId uint32

// implements sort.Interface for []BucketTuple based on score
type ByScore []BucketTuple

func (b ByScore) Len() int           { return len(b) }
func (b ByScore) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b ByScore) Less(i, j int) bool { return b[i].score < b[j].score }

var POWS []int
var PARS [][]PartitionTuple

func normalize(v []float64) []float64 {
	var vec []float64

	n := 0.0
	for _, e := range v {
		n += e * e
	}
	n = math.Pow(n, 0.5)

	for _, e := range v {
		vec = append(vec, e/n)
	}

	return vec
}

func uint32bytes(u uint32) []byte {
	bytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(bytes, u)
	return bytes
}

func getPowers() []int {
	var pows []int
	for i := 0.0; i < NR_SMAMPLE_DIM; i++ {
		pows = append(pows, int(math.Pow(2.0, i)))
	}
	return pows
}

func getPartitions() [][]PartitionTuple {
	var pars [][]PartitionTuple
	for i := 0; i < NR_PARTITIONS; i++ {
		list := rand.Perm(100)
		var par []PartitionTuple
		for _, i := range list[:10] {
			par = append(par, PartitionTuple{i, 0.0})
		}
		pars = append(pars, par)
	}
	return pars
}

func partVec(v []float64, part []PartitionTuple) PartResult {
	ind, pwi, sum := 0, 0, 0.0
	for _, ptup := range part {
		if v[ptup.dim] >= 0.0 {
			ind += POWS[pwi]
			pwi += 1
			sum += math.Abs(v[ptup.dim])
		}
	}
	return PartResult{ind, sum}
}

func getin(filename string) <-chan Vector {

	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}

	in := make(chan Vector)

	// scan line by line, parse and queue up
	go func() {
		var vid uint32 = 0
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			// Later I want to create a buffer of lines, not just line-by-line here ...
			line := scanner.Text()

			var v []float64
			// parse line to a float vector
			for _, s := range strings.Split(line, " ")[1:] {
				i, err := strconv.ParseFloat(s, 64)
				if err != nil {
					log.Fatal(err)
				} else {
					v = append(v, i)
				}
			}
			v = normalize(v)

			in <- Vector{vid, v, ""}

			if math.Mod(float64(vid), 50000) == 0 {
				fmt.Println("Building: ", vid)
			}

			vid += 1
		}

		if err := scanner.Err(); err != nil {
			log.Fatal(err)
		}

		ev := Vector{0.0, []float64{}, "DONE"}
		in <- ev

		close(in)
		file.Close()
	}()

	return in
}

func partition(pind int, ptups []PartitionTuple, in <-chan Vector, out *os.File, wg *sync.WaitGroup) {

	var buckets []Bucket
	// init buckets
	for i := 0.0; i < NR_BUCKETS; i++ {
		buckets = append(buckets, Bucket{})
	}

	bucket_size := uint64(NR_VEC) * uint64(unsafe.Sizeof(VecId))

	// partition
	go func() {
		for v := range in {
			if v.end == "DONE" {

				for bi, bk := range buckets {
					sort.Sort(ByScore(bk.tuples))
					// fill up with NULL_VAL
					var bkids []uint32
					for _, bt := range bk.tuples {
						bkids = append(bkids, bt.id)
					}

					if len(bkids) < NR_VEC {
						for i := len(bkids) - 1; i < NR_VEC; i++ {
							bkids = append(bkids, NULL_VAL)
						}
					}

					// serialize to bytes
					var bytes []byte
					for _, u := range bkids[:NR_VEC] {
						bytes = append(bytes, uint32bytes(u)...)
					}

					// write to file
					offset := uint64(pind*NR_BUCKETS)*bucket_size + uint64(bi)*bucket_size
					_, err := out.Seek(int64(offset), 0) // 0 = Beginning of file pos
					if err != nil {
						log.Fatal(err)
					}

					_, werr := out.Write(bytes)
					if werr != nil {
						log.Fatal(werr)
					}

				}

				// fmt.Println(buckets)
				fmt.Println("Done with part", pind)
				wg.Done()

			} else {
				//distribute vector for this partition
				pres := partVec(v.vector, ptups)

				if pres.score > 0.05 {
					buckets[pres.buckIdx].tuples = append(buckets[pres.buckIdx].tuples, BucketTuple{v.id, pres.score})
				}

			}
		}

	}()

}

func main() {
	// [[(98, 0.0), (54, 0.0), ..., (77, 0.0)], ... ]
	PARS = getPartitions()

	// 2^[0:NR_DIM]: [1, 2, 4, 8, ..., 512]
	POWS = getPowers()

	wg := &sync.WaitGroup{}

	var chans [NR_PARTITIONS]chan Vector

	// open output file
	out, err := os.Create(OUT)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	for i := range chans {
		chans[i] = make(chan Vector)
		partition(i, PARS[i], chans[i], out, wg)
		wg.Add(1)
	}

	inc := getin(IN)
	for line := range inc {
		for i := range chans {
			chans[i] <- line
		}
	}

	wg.Wait()
}
