#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Install these targets explicitly with rustup before building.
export IPHONEOS_DEPLOYMENT_TARGET=18.0
for target in aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin; do
  cargo build --locked --release -p flarellm-native --target "$target"
done
mkdir -p apple/Frameworks
# Replace only this script's generated bundle.
rm -rf apple/Frameworks/FlareNative.xcframework
xcodebuild -create-xcframework \
  -library "$ROOT/target/aarch64-apple-ios/release/libflare_native.a" -headers "$ROOT/apple/Headers" \
  -library "$ROOT/target/aarch64-apple-ios-sim/release/libflare_native.a" -headers "$ROOT/apple/Headers" \
  -library "$ROOT/target/aarch64-apple-darwin/release/libflare_native.a" -headers "$ROOT/apple/Headers" \
  -output "$ROOT/apple/Frameworks/FlareNative.xcframework"
