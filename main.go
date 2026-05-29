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
	"io"
	"math"
	"math/rand"
	"strconv"

	"github.com/pointlander/gradient/exp"
)

//go:embed secom.zip
var Data embed.FS

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
		}
	}
	length := len(secom)
	width := len(secom[0])
	context := exp.Context[float64]{}
	set := context.NewSet()
	set.Add("a", 2, length)
	set.Add("b", width, length)
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
			b[index] = f * .001
			index++
		}
	}

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

	loss := Avg(Quadratic(Mul(Dropout(Square(set.Get("a")), dropout), T(set.Get("b"))), T(set.Get("b"))))

	for iteration := range 1024 {
		set.Zero()
		l := exp.Gradient(loss).X[0]
		fmt.Println(iteration, l)
		set.Adam(exp.B1, exp.B2, .1)
	}
}
