package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestUpdate_FirstCallSeedsWithoutReturn(t *testing.T) {
	s := NewVolatilityStreamer()
	if err := s.Update(100.0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := s.Count(); got != 0 {
		t.Fatalf("first update should not produce a return, got count %d", got)
	}
	if got := s.RealizedVolatility(); got != 0 {
		t.Fatalf("expected zero volatility before any return, got %v", got)
	}
}

func TestUpdate_ConstantPriceHasZeroVolatility(t *testing.T) {
	s := NewVolatilityStreamer()
	for i := 0; i < 50; i++ {
		if err := s.Update(100.0); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if got := s.RealizedVolatility(); got != 0 {
		t.Fatalf("constant price stream should have zero realized volatility, got %v", got)
	}
	if got := s.Count(); got != 49 {
		t.Fatalf("expected 49 returns, got %d", got)
	}
}

func TestUpdate_KnownReturn(t *testing.T) {
	s := NewVolatilityStreamer()
	if err := s.Update(100.0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := s.Update(110.0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := math.Abs(math.Log(110.0 / 100.0))
	if got := s.RealizedVolatility(); math.Abs(got-want) > 1e-12 {
		t.Fatalf("RealizedVolatility() = %v, want %v", got, want)
	}
}

func TestUpdate_RejectsInvalidPrices(t *testing.T) {
	for _, price := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		s := NewVolatilityStreamer()
		if err := s.Update(price); err == nil {
			t.Fatalf("Update(%v) should have returned an error", price)
		}
	}
}

func TestUpdate_InvalidPriceDoesNotCorruptState(t *testing.T) {
	s := NewVolatilityStreamer()
	if err := s.Update(100.0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := s.Update(-5); err == nil {
		t.Fatalf("expected error for negative price")
	}
	if err := s.Update(105.0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := math.Abs(math.Log(105.0 / 100.0))
	if got := s.RealizedVolatility(); math.Abs(got-want) > 1e-12 {
		t.Fatalf("RealizedVolatility() = %v, want %v (rejected price should not affect state)", got, want)
	}
	if got := s.Count(); got != 1 {
		t.Fatalf("expected 1 valid return, got %d", got)
	}
}

// goldenFixture mirrors the JSON written by testdata/reference_volatility.py.
type goldenFixture struct {
	Prices                     []float64 `json:"prices"`
	ExpectedRealizedVolatility float64   `json:"expected_realized_volatility"`
}

// TestRealizedVolatility_MatchesPythonReference feeds the same price series
// used by the numpy reference implementation (testdata/reference_volatility.py)
// through the Go streamer and checks the two agree within epsilon = 1e-5. A
// small tolerance, rather than exact equality, accounts for numpy's pairwise
// summation and Go's sequential running sum accumulating floating-point error
// in a different order.
func TestRealizedVolatility_MatchesPythonReference(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "golden.json"))
	if err != nil {
		t.Fatalf("reading golden fixture: %v", err)
	}
	var fixture goldenFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("parsing golden fixture: %v", err)
	}
	if len(fixture.Prices) < 2 {
		t.Fatalf("golden fixture needs at least 2 prices, got %d", len(fixture.Prices))
	}

	s := NewVolatilityStreamer()
	for _, price := range fixture.Prices {
		if err := s.Update(price); err != nil {
			t.Fatalf("Update(%v): %v", price, err)
		}
	}

	const epsilon = 1e-5
	got := s.RealizedVolatility()
	if diff := math.Abs(got - fixture.ExpectedRealizedVolatility); diff > epsilon {
		t.Fatalf("realized volatility diverged from Python reference: got %v, want %v (diff %v > epsilon %v)",
			got, fixture.ExpectedRealizedVolatility, diff, epsilon)
	}
}

// TestUpdate_AllocatesConstantMemory proves Update itself never allocates:
// with the old b.prices = append(b.prices, price) implementation, every call
// would eventually trigger a slice growth allocation. The O(1) accumulator
// never allocates on the hot path at all.
func TestUpdate_AllocatesConstantMemory(t *testing.T) {
	s := NewVolatilityStreamer()
	if err := s.Update(100.0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	price := 100.0
	allocs := testing.AllocsPerRun(1000, func() {
		price *= 1.0001
		if err := s.Update(price); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	if allocs > 0 {
		t.Fatalf("Update should not heap-allocate per call (O(1) memory), got %.2f allocs/op", allocs)
	}
}

// TestStream_MemoryStaysConstant is the "flujo" (flood) test: it streams
// millions of prices through a single streamer and checks heap usage after
// the flood is essentially unchanged from before it. A streamer that still
// buffered every price (b.prices = append(b.prices, price)) would grow heap
// usage by tens of megabytes here; the O(1) accumulator should not.
func TestStream_MemoryStaysConstant(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping memory flood test in -short mode")
	}

	s := NewVolatilityStreamer()

	const warmup = 10_000
	const flood = 2_000_000

	price := 100.0
	for i := 0; i < warmup; i++ {
		price *= 1.00001
		if err := s.Update(price); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	for i := 0; i < flood; i++ {
		price *= 1.000001
		if err := s.Update(price); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	const maxGrowthBytes = 64 * 1024
	grown := int64(after.HeapAlloc) - int64(before.HeapAlloc)
	t.Logf("heap grew by %d bytes after streaming %d additional prices (count now %d)", grown, flood, s.Count())
	if grown > maxGrowthBytes {
		t.Fatalf("heap grew by %d bytes, want <= %d bytes for O(1) memory", grown, maxGrowthBytes)
	}
}
