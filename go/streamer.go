// Command streamer computes realized volatility from a live stream of prices.
//
// The core type, VolatilityStreamer, holds only the last observed price and a
// running sum of squared log-returns. Memory use is therefore O(1): it never
// grows as more prices are processed, unlike an implementation that buffers
// every price it has ever seen.
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// VolatilityStreamer computes the realized volatility of a price stream
// incrementally, in constant memory regardless of stream length.
//
// Realized volatility is defined as sqrt(sum(r_i^2)), where r_i is the i-th
// log-return between consecutive prices. Computing it this way only requires
// the previous price and a running sum, so no price history needs to be kept.
type VolatilityStreamer struct {
	lastPrice    float64
	hasLastPrice bool
	sumSqReturns float64
	count        int64
}

// NewVolatilityStreamer returns a streamer ready to accept prices.
func NewVolatilityStreamer() *VolatilityStreamer {
	return &VolatilityStreamer{}
}

// Update feeds the next price into the streamer. The first call only seeds
// the reference price and does not produce a return; every subsequent call
// folds one squared log-return into the running sum. Prices must be finite
// and strictly positive, as required by the log-return definition.
func (s *VolatilityStreamer) Update(price float64) error {
	if math.IsNaN(price) || math.IsInf(price, 0) || price <= 0 {
		return fmt.Errorf("streamer: invalid price %v, must be finite and > 0", price)
	}
	if s.hasLastPrice {
		logReturn := math.Log(price / s.lastPrice)
		s.sumSqReturns += logReturn * logReturn
		s.count++
	}
	s.lastPrice = price
	s.hasLastPrice = true
	return nil
}

// RealizedVolatility returns sqrt(sum of squared log-returns) observed so far.
func (s *VolatilityStreamer) RealizedVolatility() float64 {
	return math.Sqrt(s.sumSqReturns)
}

// Count returns the number of returns (price-to-price transitions) folded
// into the running sum so far.
func (s *VolatilityStreamer) Count() int64 {
	return s.count
}

func main() {
	s := NewVolatilityStreamer()
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		price, err := strconv.ParseFloat(line, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "streamer: skipping invalid input %q: %v\n", line, err)
			continue
		}
		if err := s.Update(price); err != nil {
			fmt.Fprintf(os.Stderr, "streamer: %v\n", err)
			continue
		}
		fmt.Printf("%d\t%.10f\n", s.Count(), s.RealizedVolatility())
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "streamer: reading stdin: %v\n", err)
		os.Exit(1)
	}
}
