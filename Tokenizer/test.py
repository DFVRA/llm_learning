
def partition(seq, low, high):
    pivot = seq[high]
    i = low - 1
    for j in range(low, high):
        if seq[j] <= pivot:
            i += 1
            seq[i], seq[j] = seq[j], seq[i]
    seq[i+1], seq[high] = seq[high], seq[i+1]
    return i + 1


def quick_sort(seq, low=0, high=None):
    if high == None:
        high = len(seq) - 1
    if low < high:
        pivot_id = partition(seq, low, high)
        quick_sort(seq, low, pivot_id-1)
        quick_sort(seq, pivot_id+1, high)

seq = [3, 4, 2, 1]
quick_sort(seq)
print(seq)
