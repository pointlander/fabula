// Copyright 2026 The Fabula Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"image/color"
	"io"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
	"strconv"

	"github.com/pointlander/fabula/kmeans"
	exp "github.com/pointlander/gradient"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

//go:embed secom.zip
var Data embed.FS

func euclidean[T exp.Number](a, b *exp.V[T]) *exp.V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, sizeA, sizeB := exp.NewV[T](a.S[1], b.S[1]), len(a.X), len(b.X)
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, sum := a.X[i:i+width], b.X[ii:ii+width], T(0.0)
			for j, ax := range av {
				diff := (ax - bv[j])
				sum += diff * diff
			}
			c.X = append(c.X, exp.Sqrt(sum))
		}
	}
	return c
}

// Euclidean computes the euclidean distance between all row vectors and all row vectors
func Euclidean[T exp.Number](k exp.Continuation[T], node int, a, b *exp.V[T], options ...map[string]interface{}) bool {
	width := a.S[0]
	sizeA, sizeB := len(a.X), len(b.X)
	c := euclidean(a, b)
	if k(c) {
		return true
	}
	for _, x := range a.D {
		if exp.IsInf(x) || exp.IsNaN(x) {
			fmt.Println("euclidean", a.D)
			panic(x)
		}
	}
	index := 0
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, cx, ad, bd, d := a.X[i:i+width], b.X[ii:ii+width], c.X[index], a.D[i:i+width], b.D[ii:ii+width], c.D[index]
			for j, ax := range av {
				if cx == 0 {
					continue
				}
				if exp.IsNaN((ax-bv[j])*d/cx) || exp.IsInf((ax-bv[j])*d/cx) {
					panic("blah")
				}
				if exp.IsNaN((bv[j]-ax)*d/cx) || exp.IsInf((bv[j]-ax)*d/cx) {
					panic("gah")
				}
				ad[j] += (ax - bv[j]) * d / cx
				bd[j] += (bv[j] - ax) * d / cx
			}
			index++
		}
	}
	for _, x := range a.D {
		if exp.IsInf(x) || exp.IsNaN(x) {
			fmt.Println("euclidean 2", a.D)
			panic(x)
		}
	}
	return false
}

// Cluster clusters some points
func Cluster[T exp.Number](x *exp.V[T], k int) ([]uint64, uint64) {
	type Point struct {
		Index   int
		Coord   []T
		Count   uint64
		Cluster uint64
	}
	points := make([]Point, x.S[1])
	for i := range x.S[1] {
		points[i].Index = i
		points[i].Coord = x.X[i*x.S[0] : i*x.S[0]+x.S[0]]
	}
	distribution := make([][]T, x.S[1])
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
		distribution[i] = make([]T, x.S[1])
		for ii := range points {
			distance := T(0.0)
			for iii := range points[ii].Coord {
				diff := points[i].Coord[iii] - points[ii].Coord[iii]
				distance += diff * diff
			}
			/*distance = exp.Sqrt(distance)
			if distance != 0 {
				distance = 1 / distance
			}*/
			distribution[i][ii] = exp.Exp(-distance/(2*stddev)) / exp.Sqrt(2*math.Pi*stddev)
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
	for range x.S[1] * 1024 {
		selected, total := exp.Convert[T](rng.Float64()), T(0.0)
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
	variance := func(points []Point) float64 {
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
	clusters := make([]uint64, x.S[1])
	for i := range points {
		clusters[points[i].Index] = points[i].Cluster
	}
	return clusters, uint64(index)
}

func main() {
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
		x := exp.NewV[float64](width, length)
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
		clusters, _ := Cluster(x, 2)
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
		context := exp.Context[float64]{}
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
			l := exp.Gradient(loss).X[0]
			fmt.Println(iteration, l)
			set.Adam(exp.B1, exp.B2, .05)
		}

		l0 = Add(Mul(set.Get("w0"), set.Get("input")), set.Get("b0"))
		l0(func(a *exp.V[float64]) bool {
			b = a.X
			return true
		})

	}
	rng := rand.New(rand.NewSource(1))
	context := exp.Context[float64]{}
	set := context.NewSet()
	set.Add("a", 4, length)
	set.AddData("b", 4, length)
	set.InitAdam(rng)
	for i, value := range b {
		set.ByName["b"].X[i] = value
	}

	//Inv := context.U(context.Inv)
	//Euclidean := context.B(Euclidean)
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
		l := exp.Gradient(loss).X[0]
		fmt.Println(iteration, l)
		set.Adam(exp.B1, exp.B2, .05)
	}

	a := set.ByName["a"].X
	input := make([][]float64, length)
	for i := range length {
		input[i] = make([]float64, set.ByName["a"].S[0])
		for ii := range input[i] {
			input[i][ii] = a[i*set.ByName["a"].S[0]+ii]
		}
	}
	meta := make([][]float64, length)
	for i := range meta {
		meta[i] = make([]float64, length)
	}
	for i := 0; i < 100; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), input, 2, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(meta); i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 2, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	aa := make(map[string][2]int)
	for i := range label {
		histogram := aa[label[i][0]]
		histogram[clusters[i]]++
		aa[label[i][0]] = histogram
	}
	for k, v := range aa {
		fmt.Println(k, v)
	}

	{
		fmt.Println()
		clusters, _ := Cluster(set.ByName["a"], 2)
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
		input := make([][]float64, length)
		for i := range secom {
			sum := 0.0
			input[i] = make([]float64, width)
			for ii := range secom[i] {
				f, err := strconv.ParseFloat(secom[i][ii], 64)
				if err != nil {
					panic(err)
				}
				if math.IsNaN(f) {
					f = 0
				}
				input[i][ii] = f
				sum += f
			}
			for ii := range input[i] {
				input[i][ii] /= sum
			}
		}
		meta := make([][]float64, length)
		for i := range meta {
			meta[i] = make([]float64, length)
		}
		for i := 0; i < 100; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), input, 2, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
		clusters, _, err := kmeans.Kmeans(1, meta, 2, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
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
