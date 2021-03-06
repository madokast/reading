from typing import List

"""
给定一个无序的整数数组，找到其中最长上升子序列的长度。

示例:

输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
说明:

可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
你算法的时间复杂度应该为 O(n2) 。
进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗?

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/longest-increasing-subsequence
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if nums == None or len(nums) == 0:
            return 0
        le = len(nums)
        dp = [1] * le

        for i in range(1, le):
            for j in range(i):
                if nums[i] > nums[j] and j + 2 > dp[i] and dp[i] <= dp[j]:
                    dp[i] = dp[j] + 1

        return max(dp)


if __name__ == "__main__":
    print(Solution().lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]))
