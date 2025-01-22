"""
给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，
写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
"""
from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if target < nums[0] or target > nums[-1]:
            return -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if target == nums[mid]:
                return mid
            elif target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return -1

s = Solution()
s.search([-1,0,3,5,9,12], 9) # 4