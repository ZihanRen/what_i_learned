# Data Structure & Algorithm Patterns

This document provides a summary of common data structure and algorithm patterns used to solve various programming problems.

## Sliding Window

* A technique used to process sequential data (arrays, strings) by maintaining a dynamic window that slides through the data
* Window boundaries are adjusted as needed to track relevant elements
* Converts O(n²) operations to O(n) by avoiding redundant work
* Applications:
  * Finding maximum/minimum sum subarray of fixed size
  * Finding longest substring with distinct characters
  * DNA sequence analysis
  * Maximum values in sliding windows

## Two Pointers

* Uses two pointers to traverse an iterable data structure, typically starting from different positions
* Pointers move in a coordinated manner based on specific conditions
* Useful for finding pairs of elements that satisfy certain conditions
* Applications:
  * Palindrome checking
  * Pair sum problems in sorted arrays
  * Three sum problems
  * Container with most water
  * Removing duplicates from sorted arrays

## Fast & Slow Pointers

* Uses two pointers that traverse at different speeds (often one at 1x speed, one at 2x speed)
* Primarily used to identify patterns, detect cycles, or find specific elements
* Applications:
  * Finding cycle in a linked list
  * Determining if a number is happy (sum of squares leads to 1)
  * Finding the middle element of a linked list
  * Finding the start of a cycle

## Merge Intervals

* Deals with problems involving overlapping intervals (represented by start and end times)
* Typically involves merging, inserting, or scheduling intervals
* Applications:
  * Merging overlapping intervals
  * Inserting new intervals into existing sets
  * Meeting room scheduling problems
  * Resource allocation
  * Time slot consolidation

## Subsets

* Used to explore all possible combinations of elements from a given data structure
* Often employs backtracking to build subsets incrementally
* Applications:
  * Generating all subsets of a set
  * Permutations of an array
  * Combination sums
  * Power set generation

## Backtracking

* Algorithmic technique that incrementally builds solutions and abandons (backtracks) when it determines a path cannot lead to a valid solution
* Systematically searches for all possible solutions while avoiding exhaustive brute force
* Applications:
  * N-Queens problem
  * Word search in a grid
  * Sudoku solver
  * Path finding in a maze
  * Generating permutations and combinations

## In-Place Manipulation (Linked Lists)

* Techniques to modify a linked list without using additional data structures
* Often uses pointer manipulation to achieve the desired transformation
* Applications:
  * Reversing a linked list
  * Detecting and removing cycles
  * Rearranging nodes in specific patterns
  * Merging sorted linked lists

## Hash Map

* Uses key-value pairs to store and retrieve data efficiently
* Provides O(1) average time complexity for lookups, insertions, and deletions
* Applications:
  * Finding pairs with a given sum
  * Tracking frequencies of elements
  * Implementing caches
  * Detecting duplicates

## Two Heaps

* Uses two heaps (typically a max heap and a min heap) to efficiently track elements
* Useful for problems requiring tracking median or partitioning elements
* Applications:
  * Finding median of a stream of numbers
  * Scheduling problems
  * K closest points
  * Merging K sorted lists

## Topological Sort

* Algorithm for ordering vertices in a directed graph such that for every edge (u,v), vertex u comes before v
* Used for scheduling and dependency resolution
* Applications:
  * Course scheduling
  * Build systems
  * Dependency resolution
  * Task scheduling with prerequisites

## Tree BFS (Breadth-First Search)

* Traverses a tree level by level, exploring all nodes at the current depth before moving to nodes at the next depth
* Uses a queue to track nodes to visit
* Applications:
  * Level order traversal
  * Finding shortest path in unweighted graphs
  * Connected components
  * Nearest neighbor problems

## Tree DFS (Depth-First Search)

* Traverses a tree by exploring as far as possible along each branch before backtracking
* Can be implemented recursively or using a stack
* Applications:
  * Path finding
  * Cycle detection
  * Topological sorting
  * Connected components

## Binary Search

* Efficiently finds elements in a sorted collection by repeatedly dividing the search interval in half
* Reduces time complexity from O(n) to O(log n)
* Applications:
  * Finding elements in sorted arrays
  * Finding insertion points
  * Finding peak elements
  * Range queries on sorted data

## Cyclic Sort

* Specialized sorting technique used when dealing with a range of numbers in a specific range
* Places each element at its correct position in one pass
* Applications:
  * Finding missing numbers
  * Finding duplicates
  * Array rearrangement problems

## Greedy Technique

* Makes the locally optimal choice at each stage with the hope of finding a global optimum
* Doesn't guarantee optimal solutions for all problems
* Applications:
  * Minimum spanning tree
  * Huffman coding
  * Activity selection problems
  * Coin change (in some cases)

## K-way Merge

* Technique to merge K sorted arrays or lists into a single sorted output
* Often uses a min heap to efficiently track the smallest elements
* Applications:
  * Merging K sorted lists
  * External sorting
  * Database operations on sorted data

## Matrix Operations

* Various techniques for manipulating 2D arrays, including row/column traversals, diagonal traversals, spiral traversals, rotations, transpositions, and reflections. Common patterns include:
  * Setting entire rows/columns to a value based on conditions
  * Matrix rotation (using transpose and reflection)
  * Various traversal methods (row-major, column-major, diagonal, spiral)

## Recursion

* Solving problems by breaking them down into smaller subproblems of the same type. Key components:
  * Base case definition (stopping condition)
  * Recursive relation (how to reduce problem size)
  * Common applications include tree/graph traversal, backtracking, and divide-and-conquer algorithms
  * Understanding recursion stack behavior is crucial for debugging and optimization

## Top K Elements

* Finding k largest/smallest elements in a collection efficiently using heaps:
  * For k largest elements: Use min-heap of size k (smallest element always at top)
  * For k smallest elements: Use max-heap of size k (largest element always at top)
  * Time complexity O(n log k) which is more efficient than sorting the entire array O(n log n)
  * Useful for streaming data where elements arrive sequentially

## Knowing What to Track

## Union Find

## Bitwise Manipulation

## Dynamic Programming

## Math and Geometry

## Trie
