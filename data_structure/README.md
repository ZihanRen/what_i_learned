# Data Structure & Algorithm Patterns

This document provides a summary of common data structure and algorithm patterns used to solve various programming problems.

## Sliding Window

* A technique used to process sequential data (arrays, strings) by maintaining a dynamic window that slides through the data
* Window boundaries are adjusted as needed to track relevant elements

The sliding window pattern is used to process sequential data, arrays, and strings, for example, to efficiently solve subarray or substring problems. It involves maintaining a dynamic window that slides through the array or string, adjusting its boundaries as needed to track relevant elements or characters. The window is used to slide over the data in chunks corresponding to the window size, and this can be set according to the problem's requirements. It may be viewed as a variation of the two pointers pattern, with the pointers being used to set the window bounds.

Imagine you're in a long hallway lined with paintings, and you're looking through a narrow frame that only reveals a portion of this hallway at any time. As you move the frame along the hallway, new paintings come into view while others leave the frame. This process of moving and adjusting what's visible through the frame is akin to how the sliding window technique operates over data.

Why is this method more efficient? Consider we need to find k consecutive integers with the largest sum in an array. The time complexity of the naive solution to this problem would be O(kn), because we need to compute sums of all subarrays of size 𝑘. On the other hand, if we employ the sliding window pattern, instead of computing the sum of all elements in the window, we can just subtract the element exiting the window, add the element entering the window, and update the maximum sum accordingly. In this way, we can update the sums in constant time, yielding an overall time complexity of O(n)
. To summarize, generally, the computations performed every time the window moves should take O(1) time or a slow-growing function, such as the log of a small variable.

The following illustration shows a possibility of how a window could move along an array:

![image.png](attachment:image.png)


* Converts O(n²) operations to O(n) by avoiding redundant work
* Applications:
  * Finding maximum/minimum sum subarray of fixed size
  * Finding longest substring with distinct characters
  * DNA sequence analysis
  * Maximum values in sliding windows


Questions list:
LC 187
LC 239
LC 727



## Two Pointers

* Uses two pointers to traverse an iterable data structure, typically starting from different positions
* Pointers move in a coordinated manner based on specific conditions
* Useful for finding pairs of elements that satisfy certain conditions

The two pointers pattern is a versatile technique used in problem-solving to efficiently traverse or manipulate sequential data structures, such as arrays or linked lists. As the name suggests, it involves maintaining two pointers that traverse the data structure in a coordinated manner, typically starting from different positions or moving in opposite directions. These pointers dynamically adjust based on specific conditions or criteria, allowing for the efficient exploration of the data and enabling solutions with optimal time and space complexity. Whenever there's a requirement to find two data elements in an array that satisfy a certain condition, the two pointers pattern should be the first strategy to come to mind.

The pointers can be used to iterate through the data structure in one or both directions, depending on the problem statement. For example, to identify whether a string is a palindrome, we can use one pointer to iterate the string from the beginning and the other to iterate it from the end. At each step, we can compare the values of the two pointers and see if they meet the palindrome properties.

The two pointers technique typically converts O(n²) operations to O(n) by avoiding redundant work.

#### Applications:
* Reversing arrays
* Pair with given sum in a sorted array: Given a sorted array of integers, find a pair in the array that sums to a number T.

Many problems in the real world use the two pointers pattern. Let's look at an example:

Memory management: The two pointers pattern is vital in memory allocation and deallocation. The memory pool is initialized with two pointers: the start pointer, pointing to the beginning of the available memory block, and the end pointer, indicating the end of the block. When a process or data structure requests memory allocation, the start pointer is moved forward, designating a new memory block for allocation. Conversely, when memory is released (deallocated), the start pointer is shifted backward, marking the deallocated memory as available for future allocations.

Questions list:

LC 15
LC 125


## Fast & Slow Pointers

* Uses two pointers that traverse at different speeds (often one at 1x speed, one at 2x speed)
* Primarily used to identify patterns, detect cycles, or find specific elements

Similar to the two pointers pattern, the fast and slow pointers pattern uses two pointers to traverse an iterable data structure, but at different speeds, often to identify patterns, detect cycles, or find specific elements. The speeds of the two pointers can be adjusted according to the problem statement. Unlike the two pointers approach, which is concerned with data values, the fast and slow pointers approach is often used to determine specific pattern or structure in the data.

The key idea is that the pointers start at the same location and then start moving at different speeds. Generally, the slow pointer moves forward by a factor of one, and the fast pointer moves by a factor of two. This approach enables the algorithm to detect patterns or properties within the data structure, such as cycles or intersections. If there is a cycle, the two are bound to meet at some point during the traversal. To understand the concept, think of two runners on a track. While they start from the same point, they have different running speeds. If the track is circular, the faster runner will overtake the slower one after completing a lap.

#### Example

Middle of the linked list: Given the head of a singly linked list, return the middle node of the linked list. One pointer moves at speed of 1 and other moves at speed of 2. When the second pointer reaches to the end, we get the middle value.

#### Applications:

Linear data structure: The input data can be traversed in a linear fashion, such as an array, linked list, or string.

In addition, if either of these conditions is fulfilled:

Cycle or intersection detection: The problem involves detecting a loop within a linked list or an array or involves finding an intersection between two linked lists or arrays.

Find the starting element at the second quantile: The problem involves finding the starting element of the second quantile, i.e., second half, second tertile, second quartile, etc. For example, the problem asks to find the middle element of an array or a linked list.

Questions list:
lc 202
lc 141



## Merge Intervals

* Deals with problems involving overlapping intervals (represented by start and end times)
* Typically involves merging, inserting, or scheduling intervals

The merge intervals pattern deals with problems involving overlapping intervals. Each interval is represented by a start and an end time. For example, an interval of [10,20] start at 10 and end at 20. This pattern involves tasks such as merging intersecting intervals, inserting new intervals into existing sets, or determining the minimum number of intervals needed to cover a given range. The most common problems solved using this pattern are event scheduling, resource allocation, and time slot consolidation.

The key to understanding this pattern and exploiting its power lies in understanding how any two intervals may overlap. The illustration below shows different ways in which two intervals can relate to each other:

e.g. 
* [1,4], [3,7] -> [1,7]
* [1,4], [3,7], [9,12] -> [1,7], [9,12]

Application:
1. Merge intervals: Given a sorted list of intervals, merge all overlapping intervals.
2. Meeting rooms: Given an array of meeting time intervals consisting of start and end times, determine if a person could attend all meetings.

Questions list:
lc 56
lc 57


## Subsets

* Used to explore all possible combinations of elements from a given data structure
* Often employs backtracking to build subsets incrementally

The subsets pattern is an important strategy to solve coding problems that involve exploring all possible combinations of elements from a given data structure. This pattern can be useful when dealing with sets containing unique elements or arrays/lists that may contain duplicate elements. It is used to generate all specific subsets based on the conditions that the problem provides us.

The common method used is to build the subsets incrementally, including or excluding each element of the original data structure, depending on the constraints of the problem. This process is continued for the remaining elements until all desired subsets have been generated.

The following illustration shows how subsets are made from a given array:

supposing we have [1,2,3]

split 1: [] vs [1]

split 2: 2+[], 2+[1], [], [1] -> [],[1],[2],[1,2]

split 3: 3+[], 3+[1], 3+[2], 3+[1,2], [], [1], [2], [1,2] - > [], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]

Note: We sometimes also use a programming technique known as backtracking to generate the required subsets of a given data structure of elements. Backtracking applies to a broader range of problems where exhaustive search, that is, evaluating all possibilities, is required. These problems may involve various constraints, rules, or conditions that guide the exploration process. Not all of these problems involve finding subsets. That is why it is necessary to discuss Subsets as a separate programming pattern.

The following examples illustrate some problems that can be solved with this approach:

1. Permutations: Return all possible permutations of an array of distinct integers.
2. Combination sum: Return all combinations of integers in an array that add up to a target number.

Questions list:
LC 78
LC 46


## Backtracking

* Algorithmic technique that incrementally builds solutions and abandons (backtracks) when it determines a path cannot lead to a valid solution
* Systematically searches for all possible solutions while avoiding exhaustive brute force

Backtracking is an algorithmic technique for solving problems by incrementally constructing choices to the solutions. We abandon choices as soon as it is determined that the choice cannot lead to a feasible solution. On the other side, brute-force approaches attempt to evaluate all possible solutions to select the required one. Backtracking avoids the computational cost of generating and testing all possible solutions. This makes backtracking a more efficient approach. Backtracking also offers efficiency improvements over brute-force methods by applying constraints at each step to prune non-viable paths.

The backtracking algorithm can be implemented using recursion. We use recursive calls where each call attempts to move closer towards a feasible solution. This can be outlined as follows after starting from the initial point as the current point:

1. Step 1: If the current point represents a feasible solution, declare success and terminate the search.

2. Step 2: If all paths from the current point have been explored (i.e., the current point is a dead-end) without finding a feasible solution, backtrack to the previous point.

3. Step 3: If the current point is not a dead-end, keep progressing towards the solution, and reiterate all the steps until a solution is found or all possibilities are exhausted.

#### Ideal application
Yes, if any of these conditions is fulfilled:

Complete exploration is needed for any feasible solution: The problem requires considering every possible choice to find any feasible solution.

Selecting the best feasible solution: When the goal is not just to find any feasible solution but to find the best one among all feasible solutions.

No, if the following condition is fulfilled:

Solution invalidity disqualifies other choices: In problems where failing to meet specific conditions instantly rules out all other options, backtracking might not add value.

#### Applications
Syntax analysis: In compilers, we use recursive descent parsing. It is a form of backtracking, to analyze the syntax of the program. This analysis involves matching the sequence of tokens (basic symbols of programming language) against the grammar rules of the language. When a mismatch occurs during the analysis, the parser backtracks to a previous point to try a different rule of the grammar. This ensures that even complex nested structures can be accurately understood and compiled.

Game AI (Artificial Intelligence): In games like chess or Go, AI algorithms use backtracking to try out different moves and see what happens. If a move doesn't work out well, the AI goes back and tries something else. This helps the AI learn strategies that might be better than those used by humans because it can think about lots of different moves and figure out which ones are likely to work best.

Pathfinding algorithms: In pathfinding problems like finding the way through a maze or routing in a network, backtracking is used. It tries out different paths to reach the destination. If it hits a dead end or a spot it can't pass through, it goes back and tries another path. This keeps happening until it finds a path that works and leads to the destination.

Questions list:
LC 51
LC 79


## In-Place Manipulation (Linked Lists)

* Techniques to modify a linked list without using additional data structures
* Often uses pointer manipulation to achieve the desired transformation

The in-place manipulation of a linked list pattern allows us to modify a linked list without using any additional memory. In-place refers to an algorithm that processes or modifies a data structure using only the existing memory space, without requiring additional memory proportional to the input size. This pattern is best suited for problems where we need to modify the structure of the linked list, i.e., the order in which nodes are linked together. For example, some problems require a reversal of a set of nodes in a linked list which can extend to reversing the whole linked list. Instead of making a new linked list with reversed links, we can do it in place without using additional memory.

The naive approach to reverse a linked list is to traverse it and produce a new linked list with every link reversed. The time complexity of this algorithm is O(n) while consuming O(n) extra space. How can we implement the in-place reversal of nodes so that no extra space is used? We iterate over the linked list while keeping track of three nodes: the current node, the next node, and the previous node. Keeping track of these three nodes enables us to efficiently reverse the links between every pair of nodes. This in-place reversal of a linked list works in O(n) time and consumes only O(1) space.

#### Real-world problems
Many problems in the real world use the in-place manipulation of a linked list pattern. Let's look at some examples.

File system management: File systems often use linked lists to manage directories and files. Operations such as rearranging files within a directory can be implemented by manipulating the underlying linked list in place.

Memory management: In low-level programming or embedded systems, dynamic memory allocation and deallocation often involve manipulating linked lists of free memory blocks. Operations such as merging adjacent free blocks or splitting large blocks can be implemented in place to optimize memory usage.

Questions list:
LC 206
LC 25


## Hash Map

* Uses key-value pairs to store and retrieve data efficiently
* Provides O(1) average time complexity for lookups, insertions, and deletions

A hash map, also known as a hash table, is a data structure that stores key-value pairs. It provides a way to efficiently map keys to values, allowing for quick retrieval of a value associated with a given key. Hash maps achieve this efficiency by using a hash function behind the scenes to compute an index (or hash code) for each key. This index determines where the corresponding value will be stored in an underlying array.

Below is an explanation of the staple methods of a hash map:

Insert(key, value): When a key-value pair is inserted into a hash map, the hash function computes an index based on the key. This index is used to determine the location in the hash map where the value will be stored. Because different keys may hash to the same index (collision), hash maps typically include a collision resolution strategy. Common methods include chaining or open addressing. In the average case, inserting a key-value pair takes O(1) time, assuming the hash function distributes keys uniformly across the array. However, in the worst case (when all the keys hash to the same index), insertion can take up to O(n) time, where n is the number of elements in the hash map.

Search(key): To retrieve a value from the hash map, the hash function is applied to the key to compute its index. Then, the value stored at that index is returned. In the average case, searching for a value takes O(1) time. In the worst case, it can take up to O(n) time due to resizing and handling collisions.

Remove(key): Removing a key-value pair typically involves finding the index based on the key's hash and then removing the value stored at that index. In the average case, removing a key-value pair takes O(1) time. In the worst case, it can take up to O(n) time due to resizing and handling collisions.

Questions list:


## Two Heaps

* Uses two heaps (typically a max heap and a min heap) to efficiently track elements
* Useful for problems requiring tracking median or partitioning elements

The two heaps pattern is an approach to solve problems that require us to find the median of a set of numbers or require us to perform operations that involve partitioning elements.

In this pattern, we use two heaps:
1. A min heap to store the larger half of the elements
2. A max heap to store the smaller half of the elements

The pattern maintains these heaps such that every element in the max heap is smaller than or equal to every element in the min heap. The sizes of the heaps are balanced such that they either have an equal number of elements or the max heap has at most one more element than the min heap.

This arrangement allows us to efficiently compute the median of the set of numbers at any time:
- If both heaps have the same number of elements, the median is the average of the top elements of both heaps.
- If the max heap has one more element than the min heap, the median is the top element of the max heap.

The time complexity for operations in this pattern is typically O(log n), as heap operations like insertion and deletion take O(log n) time.

#### Applications:
* Finding median of a stream of numbers
* Scheduling problems
* K closest points
* Merging K sorted lists

Questions list:


## Topological Sort

* Algorithm for ordering vertices in a directed graph such that for every edge (u,v), vertex u comes before v
* Used for scheduling and dependency resolution

Topological sorting is a technique used to organize a collection of items or tasks based on their dependencies. Imagine there is a list of tasks to complete, but some tasks can only be done after others are finished. There are many such tasks that depend on each other, or there's a specific sequence of actions that must be followed. For example, when baking a cake, there are several steps one needs to follow, and some steps depend on others. We can't frost the cake until it's baked, and we can't bake it until we've mixed the batter. To ensure that we don't frost the cake before baking it or mix the batter after preheating the oven, we need to sort the steps based on their dependencies. This is where topological sorting helps us. It figures out the correct sequence of steps to bake the cake efficiently.

The topological sort pattern is used to find valid orderings of elements that have dependencies on or priority over each other. These elements can be represented as the nodes of a graph, so in technical terms, topological sort is a way of ordering the nodes of a directed graph such that for every directed edge [a,b] from node a to node b, a comes before b in the ordering.

If we write a recipe for baking a cake, then the list of tasks goes like first mix the batter, then bake the cake, and finally, frost it. These tasks can also be organized in a graph, where each task is a node, and the dependencies between tasks are represented by directed edges.

However, if we mistakenly add an additional step in our recipe that contradicts any of the existing steps, it introduces a cycle in our graph. For example, if the recipe goes like mix the batter, frost the cake, bake the cake, and frost the cake, we can't frost a cake that hasn't been baked and can't bake a cake that's already frosted. Similarly, in a graph with cycles, we can't establish a clear order of tasks, making topological sorting impossible. Therefore, topological sorting is only applicable to directed acyclic graphs (DAGs), where tasks are organized in a logical sequence without any contradictions or cycles.

Topological sorting is crucial for converting a partial ordering to a complete ordering, especially when certain tasks have defined dependencies while others do not.

#### Applications:
* Course prerequisites: Given a list of courses, each with a prerequisite course, determine if it’s possible to complete all the courses and, if so, provide a valid order in which the courses can be taken.
* Build systems
* Dependency resolution
* Task scheduling with prerequisites

Questions list:


## Tree BFS (Breadth-First Search)

* Traverses a tree level by level, exploring all nodes at the current depth before moving to nodes at the next depth
* Uses a queue to track nodes to visit

BFS is an algorithm for traversing or searching tree or graph data structures. It explores all the nodes at the present depth prior to moving on to the nodes at the next depth level. The algorithm uses a queue data structure to keep track of the next nodes to visit.

In the context of trees, BFS visits all the nodes of a tree level by level, starting from the root node. It explores all nodes at the current depth before moving on to nodes at the next depth level. This level-order traversal is particularly useful when you need to find the shortest path between two nodes or when you want to visit all nodes close to the root before exploring deeper levels.

The time complexity of BFS is O(V + E), where V is the number of vertices and E is the number of edges in the graph. In the case of a tree with N nodes, the time complexity is O(N) since there are exactly N-1 edges in a tree. The space complexity is also O(N) in the worst case, when the tree is completely unbalanced.

#### Applications:
* Level order traversal
* Finding shortest path in unweighted graphs
* Connected components
* Nearest neighbor problems

Questions list:


## Tree DFS (Depth-First Search)

* Traverses a tree by exploring as far as possible along each branch before backtracking
* Can be implemented recursively or using a stack

DFS is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node and explores as far as possible along each branch before backtracking. It uses a stack data structure (or recursion, which implicitly uses a stack) to keep track of nodes to visit.

In the context of trees, DFS can be implemented in three main ways:
1. Pre-order: Visit the current node, then the left subtree, then the right subtree.
2. In-order: Visit the left subtree, then the current node, then the right subtree.
3. Post-order: Visit the left subtree, then the right subtree, then the current node.

The time complexity of DFS is O(V + E), where V is the number of vertices and E is the number of edges in the graph. In the case of a tree with N nodes, the time complexity is O(N) since there are exactly N-1 edges in a tree. The space complexity is also O(H) where H is the height of the tree, which could be O(N) in the worst case of a skewed tree.

#### Applications:
* Path finding
* Cycle detection
* Topological sorting
* Connected components

Questions list:


## Binary Search

* Efficiently finds elements in a sorted collection by repeatedly dividing the search interval in half
* Reduces time complexity from O(n) to O(log n)

Binary search is an efficient search algorithm for searching a target value in sorted arrays or sorted lists that support direct addressing (also known as random access). It follows a divide-and-conquer approach, significantly reducing the search space with each iteration. The algorithm uses three indexes—start, end, and middle—and proceeds as follows:

1. Set the start and end indexes to the first and last elements of the array, respectively.

2. Calculate the position of the middle index by taking the average of the start and end indexes. For example, if start=0 and end=7, then middle=⌊(0+7)/2⌋=3.

   Compare the target value with the element at the middle index.

3. If the target value is equal to the middle index element, we have found the target, and the search terminates.

4. If the target value is less than the middle index element, update the end index to middle−1 and repeat from step 2 onwards. Because the array is sorted, all the values between the middle and the end indexes will also be greater than the target value. Therefore, there's no reason to consider that half of the search space.

5. If the target value is greater than the middle index element, update the start index to middle+1 and repeat from step 2. Again, because the array is sorted, all the values between the start and the middle indexes will also be less than the target value. Therefore, there's no reason to consider that half of the search space.

Continue the process until the target value is found or if the search space is exhausted, that is, if the start index has crossed the end index. This means that the algorithm has explored all possible values, which implies that the search space is now empty and the target value is not present.

Binary search reaches the target value in O(log(n)) time because we divide the array into two halves at each step and then focus on only one of these halves. If we had opted for the brute-force approach, we would have had to traverse the entire array without any partitioning to search for the target value, which would take O(n) in the worst case.

#### Applications:
* Finding elements in sorted arrays
* Finding insertion points
* Finding peak elements
* Range queries on sorted data

Questions list:


## Cyclic Sort

this is better tutorial: https://leetcode.com/explore/learn/card/graph/623/kahns-algorithm-for-topological-sorting/3886/


* Specialized sorting technique used when dealing with a range of numbers in a specific range
* Places each element at its correct position in one pass

Cyclic sort is a simple and efficient in-place sorting algorithm used for sorting arrays with integers in a specific range, most of the time [1 – n]. It places each element in its correct position by iteratively swapping it with the element at the position where it should be located. This process continues until all elements are in their proper places, resulting in a sorted array.

But how do we know the correct position of any element? This is where the algorithm makes things even easier: the correct place of any element is simply the value of element - 1. For example, if we have the array [3,1,2], then the correct position of the first element, (3-1)th index, i.e., index 2 and not 0. Similarly, the correct position for the elements 1 and 2 is index 0 and 1, respectively.

Is there a way to determine whether to use cyclic sort on a given unsorted array of integers? The answer is: Yes. We can identify cycles within the array, which are nothing but the subsequences of numbers that are not in sorted order. A cycle occurs when there's an element that is out of place, and swapping it with the element at its correct position moves another element out of place, forming a cycle.

Unlike algorithms with nested loops or recursive calls, cyclic sort achieves a linear time complexity of O(n) due to its streamlined approach of iteratively placing each element in its correct position within a single pass through the array. This makes it fast and reliable, especially for small arrays. Moreover, cyclic sort doesn't require any extra space, as it sorts the elements in place.

Cyclic sort, while efficient in specific scenarios, does come with its limitations. Let’s go over a few:

Limited range: Cyclic sort’s efficiency is contingent upon the range of elements being limited and known beforehand. When dealing with arrays with an unknown or expansive range, the cyclic sort may not perform optimally. For example, if we have an array of prices of goods ranging from 1 to 100, cyclic sort would efficiently organize them. However, if the price range extended beyond or was unknown, cyclic sort might not provide optimal results.

Not stable: Cyclic sort lacks stability as a sorting algorithm. This means it may alter the relative order of equal elements during sorting, which could be undesirable in scenarios where maintaining the original order is important. For example, in a task queue where tasks with equal priority should be processed in the order they were received, cyclic sort might reorder tasks with the same priority, disrupting the original sequence.

Not suitable for noninteger arrays: Cyclic sort is optimized for sorting arrays of integers, attempting to use it on noninteger arrays may not produce the desired outcome. Suppose we want to sort an array of names alphabetically, then using cyclic sort is not a good choice.

Not suitable when multiple attributes play a significant role: Cyclic sort is primarily designed for arrays of integers only, so it may not handle cases where the input has multiple attributes associated with it. For example, given an array containing objects representing employees, where each object has attributes such as name, age, and salary, if we need to sort the objects with respect to all three attributes then using cyclic sort on just salary may not produce the desired outcome. This is because the other two attributes play an equal role in deciding the final order, so we can’t just take one attribute while sorting the array.


#### Demo code and time complexity
* time complexity: O(n)
* space complexity: O(1)

```
def cyclic_sort(nums):
    # initial
    i = 0
    while i < len(nums)-1:
        correct_idx = nums[i] - 1
        # keep swapping until correct_idx equals to i
        if correct_idx != i:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i+=1
    return nums
```

#### Applications:
* Finding missing numbers
* Finding duplicates
* Array rearrangement problems

Questions list:
LC 41
LC 448

## Greedy Technique

* Makes the locally optimal choice at each stage with the hope of finding a global optimum
* Doesn't guarantee optimal solutions for all problems

An algorithm is a series of steps used to solve a problem and reach a solution. In the world of problem-solving, there are various types of problem-solving algorithms designed for specific types of challenges. Among these, greedy algorithms are an approach for tackling optimization problems where we aim to find the best solution under given constraints.

Imagine being at a buffet, and we want to fill the plate with the most satisfying combination of dishes available, but there's a catch: we can only make our choice one dish at a time, and once we move past a dish, we can't go back to pick it up. In this scenario, a greedy approach would be to always choose the dish that looks most appealing to us at each step, hoping that we end up with the best possible meal.

Greedy is an algorithmic paradigm that builds up a solution piece by piece. It makes a series of choices, each time picking the option that seems best at the moment, the most greedy choice, with the goal of finding an overall optimal solution. They don't worry about the future implications of these choices and focus only on maximizing immediate benefits. This means it chooses the next piece that offers the most obvious and immediate benefit. A greedy algorithm, as the name implies, always makes the choice that seems to be the best at the time. It makes a locally-optimal choice in the hope that it will lead to a globally optimal solution. In other words, greedy algorithms are used to solve optimization problems.

Greedy algorithms work by constructing a solution from the smallest possible constituent parts. However, it's important to understand that these algorithms might not always lead us to the best solution for every problem. This is because, by always opting for the immediate benefit, we might miss out on better options available down the line. Imagine if, after picking the most appealing dishes, we realize we've filled our plate too soon and missed out on our favorite dish at the end of the buffet. That's a bit like a greedy algorithm getting stuck in what's called a local optimal solution without finding the global optimal solution or the best possible overall solution.

However, let's keep in mind that for many problems, especially those with a specific structure, greedy algorithms work wonderfully. One classic example where greedy algorithms shine is in organizing networks, like connecting computers with the least amount of cable. Prim's algorithm, for instance, is a greedy method that efficiently finds the minimum amount of cable needed to connect all computers in a network.

#### Applications:
* Minimum spanning tree
* Huffman coding
* Activity selection problems
* Coin change (in some cases)

Questions list:


## K-way Merge

* Technique to merge K sorted arrays or lists into a single sorted output
* Often uses a min heap to efficiently track the smallest elements

The K-way merge pattern is an essential algorithmic strategy for merging K sorted data structures, such as arrays and linked lists, into a single sorted data structure. This technique is an expansion of the standard merge sort algorithm, which traditionally merges two sorted data structures into one.

To understand the basics of this algorithm, first, we need to know the basic idea behind the K-way merge algorithm. The K-way merge algorithm works by repeatedly selecting the smallest (or largest, if we're sorting in descending order) element from among the first elements of the K input lists and adding this element to a new output list (with the same data type as the inputs). This process is repeated until all elements from all input lists have been merged into the output list, maintaining the sorted order.

Now, let's take a closer look at how the algorithm works. The K-way merge algorithm comprises two main approaches to achieve its goal, both leading to the same result.

Here, we'll discuss one of the approaches, which uses a minheap:

1. Insert the first element of each list into a min heap. This sets up our starting point, with the heap helping us efficiently track the smallest current element among the lists.

2. Remove the smallest element from the heap (which is always at the top) and add it to the output list. This ensures that our output list is being assembled in sorted order.

3. Keep track of which list each element in the heap came from. This is for knowing where to find the next element to add to the heap.

4. After removing the smallest element from the heap and adding it to the output list, replace it with the next element from the same list the removed element belonged to.

5. Repeat steps 2–4 until all elements from all input lists have been merged into the output list.

#### Applications:
* Merging K sorted lists
* External sorting
* Database operations on sorted data

Questions list:


## Matrix Operations

* Various techniques for manipulating 2D arrays, including row/column traversals, diagonal traversals, spiral traversals, rotations, transpositions, and reflections. Common patterns include:
  * Setting entire rows/columns to a value based on conditions
  * Matrix rotation (using transpose and reflection)
  * Various traversal methods (row-major, column-major, diagonal, spiral)

Questions list:


## Recursion

* Solving problems by breaking them down into smaller subproblems of the same type. Key components:
  * Base case definition (stopping condition)
  * Recursive relation (how to reduce problem size)
  * Common applications include tree/graph traversal, backtracking, and divide-and-conquer algorithms
  * Understanding recursion stack behavior is crucial for debugging and optimization

Questions list:


## Top K Elements

* Finding k largest/smallest elements in a collection efficiently using heaps:
  * For k largest elements: Use min-heap of size k (smallest element always at top)
  * For k smallest elements: Use max-heap of size k (largest element always at top)
  * Time complexity O(n log k) which is more efficient than sorting the entire array O(n log n)
  * Useful for streaming data where elements arrive sequentially

Questions list:


## Knowing What to Track

Questions list:

## Union Find

Questions list:

## Bitwise Manipulation

Questions list:

## Dynamic Programming

Questions list:

## Math and Geometry

Questions list:

## Trie

Questions list:


# Memory of important time complexity

* find if an element appear in a set, o(1)
* sum of total digits square ologn

