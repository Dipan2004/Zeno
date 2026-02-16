#include <iostream>

int main() {
    // Function to check if three numbers form a "threesome"
    auto is_threesome = [](int a, int b, int c) -> bool {
        return a + b == c || a + c == b || b + c == a;
    };

    // Test cases
    std::cout << (is_threesome(1, 2, 3) ? "True" : "False") << std::endl;  // False
    std::cout << (is_threesome(5, 4, 9) ? "True" : "False") << std::endl;  // True
    std::cout << (is_threesome(-2, -3, -5) ? "True" : "False") << std::endl; // False

    return 0;
}