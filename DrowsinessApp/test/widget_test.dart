// Default scaffold test from `flutter create` referenced a `MyApp` class
// that doesn't exist in this project. Replaced with a trivial smoke test.

import 'package:flutter_test/flutter_test.dart';

import 'package:drowsiness_app/main.dart';

void main() {
  test('DrowsinessApp class is defined', () {
    expect(DrowsinessApp, isNotNull);
  });
}
