#Exercise 2: Create a function that takes a list of numbers and returns a dictionary with the count, mean, median, and standard deviation.
import statistics

def compute_stats(numbers):
    stats = {}
    stats['count'] = len(numbers)
    stats['mean'] = sum(numbers) / len(numbers)
    stats['median'] = statistics.median(numbers)
    stats['std_dev'] = statistics.stdev(numbers)
    return stats

# Example usage
numbers = [2, 5, 7, 10, 15, 20, 25, 30, 35, 40]
stats = compute_stats(numbers)
print("Statistics:")
for key, value in stats.items():
    print(f"{key}: {value}")
