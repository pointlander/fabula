// Copyright 2026 The Fabula Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
	"strconv"

	"github.com/pointlander/gradient"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

//go:embed secom.zip
var Data embed.FS

var (
	// FlagFull full mode
	FlagFull = flag.Bool("full", false, "full mode")
)

// FullMode is the full mode
func FullMode() {
	file, err := Data.Open("secom.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	var secom [][]string
	var label [][]string
	for _, f := range reader.File {
		if f.Name == "secom.data" {
			input, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(input)
			reader.Comma = ' '
			secom, err = reader.ReadAll()
			if err != nil {
				panic(err)
			}
			input.Close()
		} else if f.Name == "secom_labels.data" {
			input, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(input)
			reader.Comma = ' '
			label, err = reader.ReadAll()
			if err != nil {
				panic(err)
			}
			input.Close()
		}
	}
	counta, countb := 0, 0
	for _, l := range label {
		if l[0] == "1" {
			counta++
		} else {
			countb++
		}
	}
	fmt.Println(counta, countb)
	length := len(secom)
	width := len(secom[0])
	{
		x := gradient.NewV[float64](width, length)
		for i := range length {
			for ii := range width {
				f, err := strconv.ParseFloat(secom[i][ii], 64)
				if err != nil {
					panic(err)
				}
				if math.IsNaN(f) {
					f = 0
				}
				x.X = append(x.X, f)
			}
		}
		clusters := x.ClusterKMeansPlusPlus(1, 2, 50)
		aa := make(map[string][2]int)
		for i := range clusters {
			histogram := aa[label[i][0]]
			histogram[clusters[i]]++
			aa[label[i][0]] = histogram
		}
		fmt.Println()
		for k, v := range aa {
			fmt.Println(k, v)
		}
	}
	fmt.Println()
	{
		x := gradient.NewV[float64](width, length)
		for i := range length {
			for ii := range width {
				f, err := strconv.ParseFloat(secom[i][ii], 64)
				if err != nil {
					panic(err)
				}
				if math.IsNaN(f) {
					f = 0
				}
				x.X = append(x.X, f)
			}
		}
		clusters, _ := x.ClusterPageRank(2)
		aa := make(map[string][2]int)
		for i := range clusters {
			histogram := aa[label[i][0]]
			histogram[clusters[i]]++
			aa[label[i][0]] = histogram
		}
		fmt.Println()
		for k, v := range aa {
			fmt.Println(k, v)
		}
	}
	var b []float64
	{
		context := gradient.Context[float64]{}
		set := context.NewSet()
		set.Add("w0", width, 4)
		set.AddBias("b0", 4)
		set.Add("w1", 8, width)
		set.AddBias("b1", width)
		set.AddData("input", width, length)
		rng := rand.New(rand.NewSource(1))
		set.InitAdam(rng)
		input, index := set.ByName["input"], 0
		for i := range secom {
			sum := 0.0
			start := index
			for ii := range secom[i] {
				f, err := strconv.ParseFloat(secom[i][ii], 64)
				if err != nil {
					panic(err)
				}
				if math.IsNaN(f) {
					f = 0
				}
				input.X[index] = f
				sum += f
				index++
			}
			for range width {
				input.X[start] /= sum
				start++
			}
		}
		Mul := context.B(context.Mul)
		Add := context.B(context.Add)
		Everett := context.U(context.Everett)
		Quadratic := context.B(context.Quadratic)
		Avg := context.U(context.Avg)
		l0 := Everett(Add(Mul(set.Get("w0"), set.Get("input")), set.Get("b0")))
		l1 := Add(Mul(set.Get("w1"), l0), set.Get("b1"))
		loss := Avg(Quadratic(set.Get("input"), l1))

		for iteration := range 1024 {
			set.Zero()
			l := gradient.Gradient(loss).X[0]
			fmt.Println(iteration, l)
			set.Adam(gradient.B1, gradient.B2, .05)
		}

		l0 = Add(Mul(set.Get("w0"), set.Get("input")), set.Get("b0"))
		l0(func(a *gradient.V[float64]) bool {
			b = a.X
			return true
		})

	}
	rng := rand.New(rand.NewSource(1))
	context := gradient.Context[float64]{}
	set := context.NewSet()
	set.Add("a", 4, length)
	set.AddData("b", 4, length)
	set.InitAdam(rng)
	for i, value := range b {
		set.ByName["b"].X[i] = value
	}

	//Inv := context.U(context.Inv)
	//Euclidean := context.B(context.Euclidean)
	Square := context.U(context.Square)
	Mul := context.B(context.Mul)
	Dropout := context.U(context.Dropout)
	Quadratic := context.B(context.Quadratic)
	T := context.U(context.T)
	Avg := context.U(context.Avg)

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	loss := Avg(Quadratic(Mul(Dropout(Square(set.Get("a")), dropout), T(set.Get("b"))),
		Mul(Dropout(Square(set.Get("b")), dropout), T(set.Get("a")))))

	for iteration := range 1024 {
		set.Zero()
		l := gradient.Gradient(loss).X[0]
		fmt.Println(iteration, l)
		set.Adam(gradient.B1, gradient.B2, .05)
	}

	a := set.ByName["a"].X

	{
		fmt.Println()
		clusters := set.ByName["a"].ClusterKMeansPlusPlusMeta(5, 2, 100, 100)
		aa := make(map[string][2]int)
		for i := range label {
			histogram := aa[label[i][0]]
			histogram[clusters[i]]++
			aa[label[i][0]] = histogram
		}
		for k, v := range aa {
			fmt.Println(k, v)
		}
	}

	{
		fmt.Println()
		clusters, _ := set.ByName["a"].ClusterPageRank(2)
		aa := make(map[string][2]int)
		for i := range label {
			histogram := aa[label[i][0]]
			histogram[clusters[i]]++
			aa[label[i][0]] = histogram
		}
		for k, v := range aa {
			fmt.Println(k, v)
		}
	}

	{
		a := gradient.NewV[float64](width, length)
		for i := range secom {
			for ii := range secom[i] {
				f, err := strconv.ParseFloat(secom[i][ii], 64)
				if err != nil {
					panic(err)
				}
				if math.IsNaN(f) {
					f = 0
				}
				a.X = append(a.X, f)
			}
		}
		clusters := a.ClusterKMeansPlusPlusMeta(1, 2, 100, 100)
		if clusters == nil {
			panic("clustering failed")
		}
		aa := make(map[string][2]int)
		for i := range label {
			histogram := aa[label[i][0]]
			histogram[clusters[i]]++
			aa[label[i][0]] = histogram
		}
		fmt.Println()
		for k, v := range aa {
			fmt.Println(k, v)
		}
	}

	pointsa01, pointsb01 := make(plotter.XYs, 0, 8), make(plotter.XYs, 0, 8)
	pointsa02, pointsb02 := make(plotter.XYs, 0, 8), make(plotter.XYs, 0, 8)
	pointsa12, pointsb12 := make(plotter.XYs, 0, 8), make(plotter.XYs, 0, 8)
	for i := range length {
		if label[i][0] == "1" {
			pointsa01 = append(pointsa01, plotter.XY{X: a[i*2], Y: a[i*2+1]})
			pointsa02 = append(pointsa02, plotter.XY{X: a[i*2], Y: a[i*2+2]})
			pointsa12 = append(pointsa12, plotter.XY{X: a[i*2+1], Y: a[i*2+2]})
		} else {
			pointsb01 = append(pointsb01, plotter.XY{X: a[i*2], Y: a[i*2+1]})
			pointsb02 = append(pointsb02, plotter.XY{X: a[i*2], Y: a[i*2+2]})
			pointsb12 = append(pointsb12, plotter.XY{X: a[i*2+1], Y: a[i*2+2]})
		}
	}

	{
		p := plot.New()

		p.Title.Text = "y vs x"
		p.X.Label.Text = "x"
		p.Y.Label.Text = "y"

		{
			scatter, err := plotter.NewScatter(pointsa01)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			scatter.GlyphStyle.Color = color.RGBA{B: 255, A: 255}

			p.Add(scatter)
		}

		{
			scatter, err := plotter.NewScatter(pointsb01)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			scatter.GlyphStyle.Color = color.RGBA{R: 255, A: 255}

			p.Add(scatter)
		}

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cluster01.png")
		if err != nil {
			panic(err)
		}
	}

	{
		p := plot.New()

		p.Title.Text = "z vs x"
		p.X.Label.Text = "x"
		p.Y.Label.Text = "z"

		{
			scatter, err := plotter.NewScatter(pointsa02)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			scatter.GlyphStyle.Color = color.RGBA{B: 255, A: 255}

			p.Add(scatter)
		}

		{
			scatter, err := plotter.NewScatter(pointsb02)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			scatter.GlyphStyle.Color = color.RGBA{R: 255, A: 255}

			p.Add(scatter)
		}

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cluster02.png")
		if err != nil {
			panic(err)
		}
	}

	{
		p := plot.New()

		p.Title.Text = "z vs y"
		p.X.Label.Text = "y"
		p.Y.Label.Text = "z"

		{
			scatter, err := plotter.NewScatter(pointsa12)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			scatter.GlyphStyle.Color = color.RGBA{B: 255, A: 255}

			p.Add(scatter)
		}

		{
			scatter, err := plotter.NewScatter(pointsb12)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			scatter.GlyphStyle.Color = color.RGBA{R: 255, A: 255}

			p.Add(scatter)
		}

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cluster12.png")
		if err != nil {
			panic(err)
		}
	}
}

type Point[T gradient.Number] struct {
	Index   int
	Coord   []T
	Count   uint64
	Cluster uint64
}

// ClusterPageRank clusters some points
func ClusterPageRank[T gradient.Number](a *gradient.V[T], k int) ([]uint64, uint64, []Point[T]) {
	points := make([]Point[T], a.S[1])
	for i := range a.S[1] {
		points[i].Index = i
		points[i].Coord = a.X[i*a.S[0] : i*a.S[0]+a.S[0]]
	}
	distribution := make([][]T, a.S[1])
	mean := T(0.0)
	count := T(0.0)
	stddev := T(0.0)
	for i := range distribution {
		for ii := range points {
			distance := T(0.0)
			for iii := range points[ii].Coord {
				diff := points[i].Coord[iii] - points[ii].Coord[iii]
				distance += diff * diff
			}
			mean += distance
			count++
		}
	}
	mean /= count
	for i := range distribution {
		for ii := range points {
			distance := T(0.0)
			for iii := range points[ii].Coord {
				diff := points[i].Coord[iii] - points[ii].Coord[iii]
				distance += diff * diff
			}
			diff := mean - distance
			stddev += diff * diff
		}
	}
	stddev = stddev / count

	for i := range distribution {
		distribution[i] = make([]T, a.S[1])
		for ii := range points {
			distance := T(0.0)
			for iii := range points[ii].Coord {
				diff := points[i].Coord[iii] - points[ii].Coord[iii]
				distance += diff * diff
			}
			distribution[i][ii] = gradient.Exp(-distance/(2*stddev)) / gradient.Sqrt(2*math.Pi*stddev)
		}
		sum := T(0.0)
		for _, value := range distribution[i] {
			sum += value
		}
		for ii := range distribution[i] {
			if sum == 0 {
				continue
			}
			distribution[i][ii] /= sum
		}
	}
	rng := rand.New(rand.NewSource(1))
	current := 0
	for range a.S[1] * 1024 {
		selected, total := gradient.Convert[T](rng.Float64()), T(0.0)
	outer:
		for i, value := range distribution[current] {
			total += value
			switch selected := any(selected).(type) {
			case float32:
				if selected < any(total).(float32) {
					points[i].Count++
					current = i
					break outer
				}
			case float64:
				if selected < any(total).(float64) {
					points[i].Count++
					current = i
					break outer
				}
			case complex64:
				if cmplx.Abs(complex128(selected)) < cmplx.Abs(complex128(any(total).(complex64))) {
					points[i].Count++
					current = i
					break outer
				}
			case complex128:
				if cmplx.Abs(selected) < cmplx.Abs(any(total).(complex128)) {
					points[i].Count++
					current = i
					break outer
				}
			}
		}
	}
	sort.Slice(points, func(i, j int) bool {
		return points[i].Count > points[j].Count
	})
	variance := func(points []Point[T]) float64 {
		sum := 0.0
		for i := range points {
			sum += float64(points[i].Count)
		}
		avg := sum / float64(len(points))
		v := 0.0
		for i := range points {
			diff := avg - float64(points[i].Count)
			v += diff * diff
		}
		return v / float64(len(points))
	}
	varab := variance(points)
	max, index := 0.0, 0
	for i := 1; i < len(points)-1; i++ {
		vara, varb := variance(points[0:i]), variance(points[i:len(points)])
		if diff := varab - (vara + varb); diff > max {
			max, index = diff, i
		}
	}
	centers := points[0:k]
	members := points[k:]
	for i := range members {
		max := T(0.0)
		for ii := range centers {
			distance := distribution[members[i].Index][centers[ii].Index]
			switch dist := any(distance).(type) {
			case float32:
				if dist > any(max).(float32) {
					max, members[i].Cluster = distance, uint64(ii)
				}
			case float64:
				if dist > any(max).(float64) {
					max, members[i].Cluster = distance, uint64(ii)
				}
			case complex64:
				if cmplx.Abs(complex128(dist)) > cmplx.Abs(complex128(any(max).(complex64))) {
					max, members[i].Cluster = distance, uint64(ii)
				}
			case complex128:
				if cmplx.Abs(dist) > cmplx.Abs(any(max).(complex128)) {
					max, members[i].Cluster = distance, uint64(ii)
				}
			}
		}
	}
	clusters := make([]uint64, a.S[1])
	for i := range points {
		clusters[points[i].Index] = points[i].Cluster
	}
	return clusters, uint64(index), points
}

func main() {
	flag.Parse()

	if *FlagFull {
		FullMode()
		return
	}

	file, err := Data.Open("secom.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	var secom [][]string
	var label [][]string
	for _, f := range reader.File {
		if f.Name == "secom.data" {
			input, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(input)
			reader.Comma = ' '
			secom, err = reader.ReadAll()
			if err != nil {
				panic(err)
			}
			input.Close()
		} else if f.Name == "secom_labels.data" {
			input, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(input)
			reader.Comma = ' '
			label, err = reader.ReadAll()
			if err != nil {
				panic(err)
			}
			input.Close()
		}
	}

	rng := rand.New(rand.NewSource(1))
	length, width := len(secom), len(secom[0])
	a := rng.Perm(length)
	for i := range length%33 + 1 {
		a = append(a, a[i])
	}
	context := gradient.Context[float64]{}
	results := gradient.NewV[float64](width, length)
	for s := 0; s < length; s += 33 {
		set := context.NewSet()
		set.AddBias("a", width, 33)
		set.AddData("b", width, 33)
		set.InitAdam(rng)
		for ii := range 33 {
			for iii := range width {
				f, err := strconv.ParseFloat(secom[a[s+ii]][iii], 64)
				if err != nil {
					panic(err)
				}
				if math.IsNaN(f) {
					f = 0
				}
				set.ByName["a"].X[ii*width+iii] = f * 1e-4
			}
		}
		max, result := make([]uint64, 33), gradient.NewV[float64](width, 33)
		result.X = result.X[:cap(result.X)]
		for iteration := range 16 {
			b := rng.Perm(length)
			for i := range length%33 + 1 {
				b = append(b, b[i])
			}
			for i := 0; i < length; i += 33 {
				for ii := range 33 {
					for iii := range width {
						f, err := strconv.ParseFloat(secom[b[i+ii]][iii], 64)
						if err != nil {
							panic(err)
						}
						if math.IsNaN(f) {
							f = 0
						}
						set.ByName["b"].X[ii*width+iii] = f * 1e-4
					}
				}
				drop := .3
				dropout := map[string]interface{}{
					"rng":  rng,
					"drop": &drop,
				}
				Dropout := context.U(context.DropoutMatrix)
				Square := context.U(context.Square)
				Mul := context.B(context.Mul)
				Euclidean := context.B(context.Euclidean)
				Quadratic := context.B(context.Quadratic)
				Avg := context.U(context.Avg)
				Inv := context.U(context.Inv)
				loss := Avg(Quadratic(Mul(Dropout(Square(set.Get("a")), dropout), Inv(Euclidean(set.Get("b"), set.Get("b")))),
					Mul(Dropout(Square(set.Get("b")), dropout), Inv(Euclidean(set.Get("a"), set.Get("a"))))))
				set.Zero()
				l := gradient.Gradient(loss).X[0]
				fmt.Println(iteration, l)
				set.Adam(gradient.B1, gradient.B2, 1e-1)
			}
			_, _, points := ClusterPageRank(set.ByName["a"], 2)
			sort.Slice(points, func(i, j int) bool {
				return points[i].Index < points[j].Index
			})
			for i := range points {
				if points[i].Count > max[i] {
					max[i] = points[i].Count
					copy(result.X[i*width:i*width+width], points[i].Coord)
				}
			}
		}
		results.X = append(results.X, result.X...)
	}
	clusters := results.ClusterKMeansPlusPlusMeta(1, 2, 100, 100)
	if clusters == nil {
		panic("clustering failed")
	}
	aa := make(map[string][2]int)
	for i := range label {
		histogram := aa[label[a[i]][0]]
		histogram[clusters[i]]++
		aa[label[a[i]][0]] = histogram
	}
	fmt.Println()
	for k, v := range aa {
		fmt.Println(k, v)
	}
}
