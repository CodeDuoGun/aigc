"""
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 
子串
 的长度。

 

示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
"""
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        i=0
        j=1
        max_len = j-i
        while i<=j and j<=len(s)-1:
            print(i,j, s[i:j])
            if len(s)==2:
                if s[j]!=s[i]:return len(s) 
                else: return max_len
            while s[j] in s[i:j]:
                max_len = max(max_len, j-i)
                i+=1
            #     j+=1
            # else:
            j+=1
 
        return max(max_len, j-i)
    
    def leco(self, s: str)-> int:
        # 哈希集合，记录每个字符是否出现过
        occ = set()
        n = len(s)
        # 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        rk, ans = -1, 0
        for i in range(n):
            if i != 0:
                # 左指针向右移动一格，移除一个字符
                occ.remove(s[i - 1])
            while rk + 1 < n and s[rk + 1] not in occ:
                # 不断地移动右指针
                occ.add(s[rk + 1])
                rk += 1
            # 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = max(ans, rk - i + 1)
        return ans

s = Solution()
print(s.lengthOfLongestSubstring("pwwkew"))
