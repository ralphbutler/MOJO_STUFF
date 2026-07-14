def greet(name: String) -> String:
    return "Hello, " + name + "!"


def main():
    print(greet("Mojo 🔥"))

    var total = 0
    for i in range(1, 11):
        total += i
    print("sum of 1..10 =", total)

    # SIMD: square 4 floats in one operation
    var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    print("v * v =", v * v)
