# Data Structure & Algorithm Questions

This document provides a collection of algorithm questions organized by the pattern they utilize.

## Sliding Window

#### Question 1
Given a string, s, that represents a DNA subsequence, and a number k, return all the contiguous subsequences (substrings) of length 𝑘 that occur more than once in the string. The order of the returned subsequences does not matter. If no repeated substring is found, the function should return an empty set.

The DNA sequence is composed of a series of nucleotides abbreviated as A,B,C,.... For example, ACGAATTCCG is a DNA sequence. When studying DNA, it is useful to identify repeated sequences in it.

#### Question 2
Given an integer list, nums, find the maximum values in all the contiguous subarrays (windows) of size w.

#### Question 3
Given two strings, str1 and str2, find the shortest substring in str1 such that str2 is a subsequence of that substring.

A substring is defined as a contiguous sequence of characters within a string. A subsequence is a sequence that can be derived from another sequence by deleting zero or more elements without changing the order of the remaining elements.

## Two Pointers

#### Question 1
Write a function that takes a string, s, as an input and determines whether or not it is a palindrome.

Note: A palindrome is a word, phrase, or sequence of characters that reads the same backward as forward.
The string s will contain English uppercase and lowercase letters, digits, and spaces.

#### Question 2
Given an array of integers, nums, and an integer value, target, determine if there are any three integers in nums whose sum is equal to the target, that is, nums[i] + nums[j] + nums[k] == target. Return TRUE if three such integers exist in the array. Otherwise, return FALSE.

Note: A valid triplet consists of elements with distinct indexes.

## Fast & Slow Pointers

#### Question 1
Given a linked list, determine if it has a cycle in it.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where the tail connects to. If pos is -1, then there is no cycle in the linked list.

#### Question 2
Write a function that determines if a number n is happy.

A happy number is a number defined by the following process:
- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
- Those numbers for which this process ends in 1 are happy numbers.

## Merge Intervals

#### Question 1
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

#### Question 2
Given a set of non-overlapping intervals and a new interval, insert the interval at the correct position and merge all necessary intervals to produce a set of non-overlapping intervals.

## Cyclic Sort

#### Question 1
You are given an array of n integers in the range 1 to n. Each integer appears exactly once, except for one which is missing. Find and return the missing integer.

#### Question 2
You are given an array of integers containing n + 1 integers where each integer is in the range [1, n]. There is only one duplicate number in the array, return this duplicate number.

## In-Place Reversal of a Linked List

#### Question 1
Given the head of a singly linked list, reverse the list and return the reversed list.

#### Question 2
Given a linked list, reverse the nodes of the linked list k at a time and return the modified list. k is a positive integer that is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k, the remaining nodes should remain in their original order.

## Tree Breadth-First Search

#### Question 1
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

#### Question 2
Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

## Tree Depth-First Search

#### Question 1
Given the root of a binary tree, return the inorder traversal of its nodes' values.

#### Question 2
Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

## Two Heaps

#### Question 1
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

Design a data structure that supports the following two operations:
- void addNum(int num) - Add an integer number from the data stream to the data structure.
- double findMedian() - Return the median of all elements so far.

#### Question 2
Design a class to find the kth largest element in a stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.

## Subsets

#### Question 1
Given an integer array nums, return all possible subsets (the power set). The solution set must not contain duplicate subsets.

#### Question 2
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

## Binary Search

#### Question 1
Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

#### Question 2
Given a sorted array of integers and a target integer, find the first and last position of the target in the array. If the target is not found, return [-1, -1].

## Topological Sort

#### Question 1
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

#### Question 2
Given a list of tickets where tickets[i] = [fromi, toi] represent a ticket from fromi to toi, find the itinerary that uses all tickets once and returns to the starting airport. You may assume all tickets form at least one valid itinerary.

## Backtracking

#### Question 1
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other. Given an integer n, return all distinct solutions to the n-queens puzzle. Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

#### Question 2
Given an m x n grid of characters board and a string word, return true if word exists in the grid. The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

## Dynamic Programming

#### Question 1
Given a string s, find the longest palindromic substring in s.

#### Question 2
Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

## Greedy Techniques

#### Question 1
Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position. Determine if you are able to reach the last index.

#### Question 2
Given an array of intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

## K-way Merge

#### Question 1
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

#### Question 2
Given an n x n matrix where each row and column is sorted in ascending order, return the kth smallest element in the matrix.

## Graph Algorithms

#### Question 1
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]. Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

#### Question 2
There are n computers numbered from 0 to n - 1 connected by ethernet cables connections forming a network where connections[i] = [a, b] represents a connection between computers a and b. Any computer can reach any other computer directly or indirectly through the network.

Given an initial computer network connections. You can extract certain cables between two directly connected computers, and place them between any pair of disconnected computers to make them directly connected.

Return the minimum number of times you need to do this in order to make all the computers connected. If it is not possible, return -1.
