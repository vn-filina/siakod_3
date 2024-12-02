import time
import numpy as np
import matplotlib.pyplot as plt


class PriorityQueueHeap:
    def __init__(self):
        self.array = []

    def insert(self, x, p):
        self.array.append((x, p))
        self.heap_up(len(self.array) - 1)

    def extract_max(self):
        if not self.array:
            raise Exception("Priority queue is empty")
        max_element = self.array[0][0]
        self.array[0] = self.array[-1]
        self.array.pop()
        if self.array:
            self.heap_down(0)
        return max_element

    def heap_up(self, index):
        parent_index = (index - 1) // 2
        if index > 0 and self.array[index][1] > self.array[parent_index][1]:
            self.array[index], self.array[parent_index] = self.array[parent_index], self.array[index]
            self.heap_up(parent_index)

    def heap_down(self, index):
        largest = index
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        if left_child < len(self.array) and self.array[left_child][1] > self.array[largest][1]:
            largest = left_child
        if right_child < len(self.array) and self.array[right_child][1] > self.array[largest][1]:
            largest = right_child
        if largest != index:
            self.array[index], self.array[largest] = self.array[largest], self.array[index]
            self.heap_down(largest)


class ListNode:
    def __init__(self, value, priority):
        self.value = value
        self.priority = priority
        self.next = None


class PriorityQueueLazyList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert(self, x, p):
        new_node = ListNode(x, p)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def extract_max(self):
        if not self.head:
            raise Exception("Priority queue is empty")

        max_node = self.head
        max_node_prev = None
        current = self.head
        prev = None

        while current:
            if current.priority > max_node.priority:
                max_node = current
                max_node_prev = prev
            prev = current
            current = current.next

        if max_node_prev is None:
            self.head = self.head.next
        else:
            max_node_prev.next = max_node.next

        if max_node == self.tail:
            self.tail = max_node_prev

        return max_node.value


def count_time(queue_class, num_operations):
    queue = queue_class()
    trials = 10

    for i in range(num_operations):
        queue.insert(i, np.random.randint(0, num_operations))

    insert_start = time.time_ns()
    for _ in range(trials):
        queue.insert(1, np.random.randint(0, num_operations))
    insert_end = time.time_ns()

    extract_start = time.time_ns()
    for _ in range(trials):
        queue.extract_max()
    extract_end = time.time_ns()

    avg_insert_time = (insert_end - insert_start) / trials
    avg_extract_time = (extract_end - extract_start) / trials

    return avg_insert_time, avg_extract_time


queue_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
heap_insert_times = []
heap_extract_times = []
list_insert_times = []
list_extract_times = []

for size in queue_sizes:
    heap_ins_time, heap_ext_time = count_time(PriorityQueueHeap, size)
    heap_insert_times.append(heap_ins_time)
    heap_extract_times.append(heap_ext_time)

    list_ins_time, list_ext_time = count_time(PriorityQueueLazyList, size)
    list_insert_times.append(list_ins_time)
    list_extract_times.append(list_ext_time)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(queue_sizes, heap_insert_times, label="Вставка в кучу (O(log n))", marker="o")
plt.plot(queue_sizes, list_insert_times, label="Вставка в список (O(1))", marker="o")
plt.xlabel("Размер очереди")
plt.ylabel("Среднее время на вставку (нс)")
plt.title("Операция вставки")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(queue_sizes, heap_extract_times, label="Извлечение из кучи (O(log n))", marker="o")
plt.plot(queue_sizes, list_extract_times, label="Извлечение из списка (O(n))", marker="o")
plt.xlabel("Размер очереди")
plt.ylabel("Среднее время на извлечение (нс)")
plt.title("Операция извлечения")
plt.legend()

plt.tight_layout()
plt.show()