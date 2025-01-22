class Node:
    def __init__(self, data, tail=None):
        from room_business.room_data import Speech
        self.data: Speech = data
        self.tail = tail

    def last(self):
        node = self.tail
        result = self
        while node is not None:
            result = node
            node = node.tail
        return result

    def insert_node(self, node):
        node.last().tail = self.tail
        self.tail = node

    def insert_data(self, data):
        node = Node(data, tail=self.tail)
        self.insert_node(node)



class LinkedList:
    def __init__(self):
        self.head: Node = None

    def set_head(self, node: Node):
        self.head = node

    def count(self):
        node = self.head
        count = 0
        while node is not None:
            count += 1
            node = node.tail
        return count

    def last(self):
        node = self.head
        result = node
        while node is not None:
            result = node
            node = node.tail
        return result

    def append_node(self, node):
        # TODO： 如果只有一个节点，会不会有问题
        last_node = self.last()
        if last_node is not None:
            self.last().tail = node
        # 下面是一个改动参考
        # if self.head is None:  
        #     self.head = node  
        # else:  
        #     last_node = self.last()  
        #     last_node.tail = node  

    def append_data(self, data):
        node = Node(data)
        self.append_node(node)