def leftbinsearch(nums,target):
    #二分查找找到第一个所需元素
    left=0
    right=len(nums)-1
    while left<right:
        mid=(left+right)>>1
        if nums[mid]<target:
            left=mid+1#主要修改了这里
        else:
            right=mid
    return left
        
def rightbinsearch(nums,target):
    #二分查找找到最后一个所需元素
    left=0
    right=len(nums)-1
    while left<right:
        mid=(left+right+1)>>1#和上面有一点点不同left+right+1
        if nums[mid]>target:
            right=mid-1
        else:
            left=mid
    return right