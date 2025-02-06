"""给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。
"""
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 != 0:
            return False
        stack = []
        mapping = {')': '(', ']': '[', '}': '{'}
        for bracket in s:
            if not stack:
                stack.append(bracket)
            elif bracket in mapping and mapping.get(bracket) == stack[-1]:
                stack.pop()
            else:
                stack.append(bracket)
        return not stack

        