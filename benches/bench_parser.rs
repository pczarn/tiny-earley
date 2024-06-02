#![feature(test)]

use test::Bencher;
use tiny_earley::calc_recognizer;

extern crate test;
extern crate tiny_earley;

#[bench]
fn bench_parser(bench: &mut Bencher) {
    let recognizer = calc_recognizer();
    bench.bytes = 9;
    bench.iter(|| {
        let mut parser = recognizer.clone();
        parser.parse("1.0 + 2.0")
    });
}

#[bench]
fn bench_parser2(bench: &mut Bencher) {
    let recognizer = calc_recognizer();
    bench.bytes = 76;
    bench.iter(|| {
        let mut parser = recognizer.clone();
        parser.parse("1.0 + 2.0 * 3.0 + 1.0 + 2.0 * 3.0 + 1.0 + 2.0 * 3.0 / 1.0 + 2.0 * 3.01234234")
    });
}

#[bench]
fn bench_parser3(bench: &mut Bencher) {
    let recognizer = calc_recognizer();
    bench.bytes = 68 + 92 + 74;
    bench.iter(|| {
        let mut parser = recognizer.clone();
        parser.parse("1.0 + (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 \
        * 3.01234234 + (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 * 3.01234234 + \
        (2.0 * 3.0 + (1.0 + 2.0 * 3.0) + 1.0) + 2.0 * 3.0 / 1.0 + 2.0 * 3.01234234")
    });
}
