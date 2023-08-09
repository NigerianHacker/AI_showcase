from abc import ABC, abstractmethod


class Node(ABC):
    def __init__(self,val):
        self.l = None
        self.r = None
        self.v = val

    @abstractmethod
    def getInfo(self):
        pass


class IntNode(Node):
    def __init__(self, val):
        super(IntNode, self).__init__(val)
        if type(val) == int:
            self.v = val
        else:
            self.v = None
            print("Datatype must be int")

    def getInfo(self):
        pass


class StrNode(Node):
    def __init__(self, val):
        super(StrNode, self).__init__(val)
        if type(val) == str:
            self.v = val
        else:
            self.v = None
            print("Datatype must be str")

    def getInfo(self):
        pass

    def countvowels(self,string):
         num_vowels=0
         for char in string:
             if char in "aeiouAEIOU":
                 num_vowels = num_vowels+1
         return num_vowel


class BinTree:
    def __init__(self, root=None):
        self.root = root

    def insert(self, val):
        if type(val) == int:
            if self.root is None:
                self.root = IntNode(val)
            else:
                self._insert(val, self.root)
        if type(val) == str:
            if self.root is None:
                self.root = StrNode(val)
            else:
                self._strinsert(val, self.root)
    
    def countvowels(self,string):
         num_vowels=0
         for char in string:
             if char in "aeiouAEIOU":
                 num_vowels = num_vowels+1
         return num_vowels

    def _strinsert(self,val,node):
        vowel_count = self.countvowels(val)
        root_count = self.countvowels(node.v)
        if val == node.v:
            pass
        if vowel_count <= root_count:
            if node.l is not None:
                self._strinsert(val, node.l)
            else:
                node.l = StrNode(val)
        else:
            if node.r is not None:
                self._strinsert(val, node.r)
            else:
                node.r = StrNode(val)

    def _insert(self, val, node):
        if val == node.v:
            pass
        elif val < node.v:
            if node.l is not None:
                self._insert(val, node.l)
            else:
                node.l = IntNode(val)
        else:
            if node.r is not None:
                self._insert(val, node.r)
            else:
                node.r = IntNode(val)