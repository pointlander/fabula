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
	"math/rand"
	"strconv"

	"github.com/pointlander/gradient/exp"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

//go:embed secom.zip
var Data embed.FS

// Euclidean computes the euclidean distance between all row vectors and all row vectors
func Euclidean[T exp.Number](k exp.Continuation[T], node int, a, b *exp.V[T], options ...map[string]interface{}) bool {
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
	context := exp.Context[float64]{}
	set := context.NewSet()
	set.Add("a", 2, length)
	set.AddData("b", width, length)
	rng := rand.New(rand.NewSource(1))
	set.InitAdam(rng)

	b, index := set.ByName["b"].X, 0
	for i := range secom {
		for ii := range secom[i] {
			f, err := strconv.ParseFloat(secom[i][ii], 64)
			if err != nil {
				panic(err)
			}
			if math.IsNaN(f) {
				f = 0
			}
			b[index] = f * .1
			index++
		}
	}

	Inv := context.U(context.Inv)
	Euclidean := context.B(Euclidean)
	Square := context.U(context.Square)
	Mul := context.B(context.Mul)
	Dropout := context.U(context.Dropout)
	Quadratic := context.B(context.Quadratic)
	//T := context.U(context.T)
	Avg := context.U(context.Avg)

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	loss := Avg(Quadratic(Mul(Dropout(Square(set.Get("a")), dropout), Inv(Euclidean(set.Get("b"), set.Get("b")))),
		Inv(Euclidean(set.Get("b"), set.Get("b")))))

	for iteration := range 33 {
		set.Zero()
		l := exp.Gradient(loss).X[0]
		fmt.Println(iteration, l)
		set.Adam(exp.B1, exp.B2, .05)
	}

	a := set.ByName["a"].X
	pointsa, pointsb := make(plotter.XYs, 0, 8), make(plotter.XYs, 0, 8)
	for i := range length {
		if label[i][0] == "1" {
			pointsa = append(pointsa, plotter.XY{X: a[i*2], Y: a[i*2+1]})
		} else {
			pointsb = append(pointsb, plotter.XY{X: a[i*2], Y: a[i*2+1]})
		}
	}
	p := plot.New()

	p.Title.Text = "y vs x"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	{
		scatter, err := plotter.NewScatter(pointsa)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = color.RGBA{B: 255, A: 255}

		p.Add(scatter)
	}

	{
		scatter, err := plotter.NewScatter(pointsb)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = color.RGBA{R: 255, A: 255}

		p.Add(scatter)
	}

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cluster.png")
	if err != nil {
		panic(err)
	}
}
