双指针操作
    def getmidpointer(self,head):#用开满指针，由追及问题的解知道mid在中间
        if head is None:
            return head
        fast=head
        slow=head
        while fast.next and fast.next.next:
            slow=slow.next
            fast=fast.next.next
        return slow
