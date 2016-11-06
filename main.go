package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"unsafe"
)

const (
	IN                        = "sampled"
	OUT                       = "out.par"
	NR_DIM                    = 100        // number of dimensions in embedding vector space
	NR_SMAMPLE_DIM            = 10         // number of dimensions form NR_DIM to take as a sample
	NR_VEC                    = 1024       // number of vectors to store in a bucket
	NR_BUCKETS                = 1024       // number of buckets stored in a partition
	NR_PARTITIONS             = 512        // number of partitions
	NR_LINE_GOROUTINES        = 512        // number of partitions
	NULL_VAL           uint32 = 4294967295 // (2^32)-1
)

type Vector struct {
	id     uint32
	vector []float64
}

type Line struct {
	id   uint32
	line string
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

// implements sort.Interface for []BucketTuple based on score
type ByScore []BucketTuple

func (b ByScore) Len() int           { return len(b) }
func (b ByScore) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b ByScore) Less(i, j int) bool { return b[i].score < b[j].score }

var POWS []int
var PARS [][]PartitionTuple

var VecId uint32

var lineChannel = make(chan Line, 1024*NR_LINE_GOROUTINES)
var vectChannel = make(chan Vector, 1024*NR_LINE_GOROUTINES)

var prod_wg sync.WaitGroup
var cons_wg sync.WaitGroup
var part_wg sync.WaitGroup

func normalize(v []float64) []float64 {
	var vec []float64
	if len(v) == 0 {
		return vec
	}

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

func parseToVec(line string) []float64 {
	var v []float64
	s := strings.Split(line, "\t")[1]
	err := json.Unmarshal([]byte(s), &v)
	if err != nil {
		log.Fatal(err)
	}
	return normalize(v)
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

// should run till either in or done are closed
func partition(in chan Vector, pind int, ptups []PartitionTuple, out *os.File) {

	defer part_wg.Done()

	var buckets []Bucket
	// init buckets
	for i := 0.0; i < NR_BUCKETS; i++ {
		buckets = append(buckets, Bucket{})
	}

	bucket_size := uint64(NR_VEC) * uint64(unsafe.Sizeof(VecId))
	// partition

	for v := range in {
		//distribute vector for this partition
		pres := partVec(v.vector, ptups)
		if pres.score > 0.05 {
			buckets[pres.buckIdx].tuples = append(buckets[pres.buckIdx].tuples, BucketTuple{v.id, pres.score})
		}
	}

	var part_wg sync.WaitGroup

	for bi, bk := range buckets {
		part_wg.Add(1)
		go func() {
			defer part_wg.Done()
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
		}()
	}
	part_wg.Wait()
	fmt.Println("Done with part", pind)
	return
}

func getIn() {
	file, err := os.Open(IN)
	if err != nil {
		panic(err)
	}
	var i uint32 = 0
	go func() {
		defer file.Close()
		defer prod_wg.Done()

		scanner := bufio.NewScanner(bufio.NewReaderSize(file, 1024*1024*10))
		for scanner.Scan() {
			lineChannel <- Line{i, scanner.Text()}
			if math.Mod(float64(i), 50000) == 0 {
				fmt.Println("Streaming: ", i)
			}
			i += 1
		}
	}()
}

// reads lines from in, parses them to Vector and sends to out
// until either in or done is closed.
func parseLine() {
	defer cons_wg.Done()
	for l := range lineChannel {
		vectChannel <- Vector{l.id, parseToVec(l.line)}
	}
}

func main() {
	// open output file
	out, err := os.Create(OUT)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	// [[(98, 0.0), (54, 0.0), ..., (77, 0.0)], ... ]
	PARS = getPartitions()

	// 2^[0:NR_DIM]: [1, 2, 4, 8, ..., 512]
	POWS = getPowers()

	partChans := make([]chan Vector, NR_LINE_GOROUTINES)
	for i, _ := range partChans {
		// The size of the channels buffer controls how far behind the recievers
		// of the fanOut channels can lag the other channels.
		partChans[i] = make(chan Vector, 1024*NR_LINE_GOROUTINES) //, lag
		part_wg.Add(1)
		go partition(partChans[i], i, PARS[i], out)
	}

	prod_wg.Add(1)
	go getIn()

	for c := 0; c < NR_LINE_GOROUTINES; c++ {
		cons_wg.Add(1)
		go parseLine()
	}

	go func() {
		prod_wg.Wait()
		close(lineChannel)
	}()

	go func() {
		cons_wg.Wait()
		close(vectChannel)
	}()

	part_wg.Add(1)
	go func() {
		for v := range vectChannel {
			for i, _ := range partChans {
				partChans[i] <- v
			}
		}

		for i, _ := range partChans {
			close(partChans[i])
		}

		part_wg.Done()
	}()

	part_wg.Wait()
}
