def two_sum(nums, target):
    """
    Finds two numbers such that they add up to the target.
    
    Args:
    nums: List of integers.
    target: Integer representing the desired sum.

    Returns:
    A tuple containing the indices of the two numbers such that they add up to the target.
    The indices are 0-based. If no such numbers exist, returns (-1, -1).
    """
    num_to_index = {}
    
    for index, value in enumerate(nums):
        complement = target - value
        if complement in num_to_index:
            return (num_to_index[complement], index)
        num_to_index[value] = index
    
    return (-1, -1)

# Example usage and testing the function
if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(result)  # Expected output: (0, 1)