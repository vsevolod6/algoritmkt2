from collections import defaultdict, deque
from typing import List, Set, Dict, Optional, Tuple
import graphviz
import time

class DependencyGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.components = set()
        self._topological_order = None
        self._has_cycle = None
        
    def add_component(self, name: str) -> None:
        """Добавляет вершину в граф"""
        self.components.add(name)
        if name not in self.graph:
            self.graph[name] = []
        self._invalidate_cache()
    
    def add_dependency(self, from_component: str, to_component: str) -> None:
        """Добавляет ориентированное ребро зависимости"""
        self.add_component(from_component)
        self.add_component(to_component)
        self.graph[from_component].append(to_component)
        self._invalidate_cache()
    
    def _invalidate_cache(self) -> None:
        """Сбрасывает кэш результатов"""
        self._topological_order = None
        self._has_cycle = None
    
    def is_acyclic(self) -> bool:
        """Проверяет, является ли граф ацикличным"""
        if self._has_cycle is not None:
            return not self._has_cycle
        
        try:
            self.get_topological_order()
            return True
        except ValueError:
            return False
    
    def get_topological_order(self) -> List[str]:
        """Возвращает топологический порядок или вызывает исключение при наличии циклов"""
        if self._topological_order is not None:
            return self._topological_order
        
        # Алгоритм Кана для топологической сортировки
        in_degree = {node: 0 for node in self.components}
        
        # Вычисляем полустепени захода
        for node in self.components:
            for neighbor in self.graph[node]:
                in_degree[neighbor] += 1
        
        # Очередь вершин с нулевой полустепенью захода
        queue = deque([node for node in self.components if in_degree[node] == 0])
        topological_order = []
        
        while queue:
            current = queue.popleft()
            topological_order.append(current)
            
            for neighbor in self.graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(topological_order) != len(self.components):
            self._has_cycle = True
            raise ValueError("Обнаружен цикл в графе зависимостей")
        
        self._topological_order = topological_order
        self._has_cycle = False
        return topological_order
    
    def get_reverse_graph(self) -> Dict[str, List[str]]:
        """Возвращает обратный граф (для поиска зависимостей)"""
        reverse_graph = defaultdict(list)
        for node in self.components:
            for neighbor in self.graph[node]:
                reverse_graph[neighbor].append(node)
        return reverse_graph

class DependencyAnalyzer:
    def __init__(self, graph: DependencyGraph):
        self.graph = graph
        self._dfs_cache = {}
    
    def find_dependencies_bfs(self, start: str) -> List[List[str]]:
        """
        Находит все зависимости через BFS, группируя по уровням
        """
        if start not in self.graph.components:
            return []
        
        reverse_graph = self.graph.get_reverse_graph()
        visited = set()
        result = []
        queue = deque([start])
        visited.add(start)
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                current = queue.popleft()
                
                for dependency in reverse_graph.get(current, []):
                    if dependency not in visited:
                        visited.add(dependency)
                        queue.append(dependency)
                        current_level.append(dependency)
            
            if current_level:
                result.append(current_level)
        
        return result
    
    def find_dependencies_dfs(self, start: str) -> Set[str]:
        """
        Находит все уникальные зависимости через DFS
        """
        if start in self._dfs_cache:
            return self._dfs_cache[start]
        
        if start not in self.graph.components:
            return set()
        
        reverse_graph = self.graph.get_reverse_graph()
        visited = set()
        
        def dfs(node: str):
            if node in visited:
                return
            visited.add(node)
            for dependency in reverse_graph.get(node, []):
                dfs(dependency)
        
        dfs(start)
        # Убираем стартовый компонент из результата
        result = visited - {start}
        self._dfs_cache[start] = result
        return result
    
    def find_all_dependencies_dfs(self, start: str) -> Set[str]:
        """
        Полный DFS обход всех путей зависимостей
        """
        if start not in self.graph.components:
            return set()
        
        reverse_graph = self.graph.get_reverse_graph()
        all_dependencies = set()
        
        def dfs_all_paths(node: str, path: List[str]):
            for dependency in reverse_graph.get(node, []):
                if dependency not in all_dependencies:
                    all_dependencies.add(dependency)
                    dfs_all_paths(dependency, path + [dependency])
        
        dfs_all_paths(start, [start])
        return all_dependencies

class AdvancedDependencyAnalyzer(DependencyAnalyzer):
    def __init__(self, graph: DependencyGraph, weights: Optional[Dict[Tuple[str, str], float]] = None):
        super().__init__(graph)
        self.weights = weights or {}
        self._critical_path_cache = {}
    
    def add_weight(self, from_component: str, to_component: str, weight: float) -> None:
        """Добавляет вес ребру зависимости"""
        self.weights[(from_component, to_component)] = weight
    
    def find_critical_path(self, start: str) -> Tuple[List[str], float]:
        """
        Находит критический путь (самый длинный путь) от start до любого компонента
        """
        if start in self._critical_path_cache:
            return self._critical_path_cache[start]
        
        # Используем алгоритм для поиска самого длинного пути в DAG
        topological_order = self.graph.get_topological_order()
        dist = {node: float('-inf') for node in self.graph.components}
        dist[start] = 0
        predecessor = {}
        
        for node in topological_order:
            if dist[node] != float('-inf'):
                for neighbor in self.graph.graph[node]:
                    weight = self.weights.get((node, neighbor), 1.0)  # вес по умолчанию 1.0
                    if dist[neighbor] < dist[node] + weight:
                        dist[neighbor] = dist[node] + weight
                        predecessor[neighbor] = node
        
        # Находим узел с максимальным расстоянием
        max_node = max(dist, key=dist.get)
        max_dist = dist[max_node]
        
        # Восстанавливаем путь
        path = []
        current = max_node
        while current != start:
            path.append(current)
            current = predecessor.get(current)
            if current is None:
                break
        path.append(start)
        path.reverse()
        
        result = (path, max_dist)
        self._critical_path_cache[start] = result
        return result

class GraphVisualizer:
    @staticmethod
    def visualize(graph: DependencyGraph, filename: str = "dependency_graph") -> None:
        """Визуализирует граф с помощью Graphviz"""
        dot = graphviz.Digraph(comment='Dependency Graph')
        
        # Добавляем вершины
        for component in graph.components:
            dot.node(component)
        
        # Добавляем рёбра
        for from_comp in graph.graph:
            for to_comp in graph.graph[from_comp]:
                dot.edge(from_comp, to_comp)
        
        # Сохраняем и отображаем
        dot.render(filename, format='png', cleanup=True)
        print(f"Граф сохранен как {filename}.png")
    
    @staticmethod
    def visualize_with_critical_path(graph: DependencyGraph, critical_path: List[str], 
                                   filename: str = "critical_path") -> None:
        """Визуализирует граф с выделенным критическим путём"""
        dot = graphviz.Digraph(comment='Critical Path')
        
        # Добавляем вершины
        for component in graph.components:
            if component in critical_path:
                dot.node(component, color='red', style='filled', fillcolor='lightcoral')
            else:
                dot.node(component)
        
        # Добавляем рёбра
        for from_comp in graph.graph:
            for to_comp in graph.graph[from_comp]:
                edge_color = 'red' if (from_comp in critical_path and to_comp in critical_path and 
                                     critical_path.index(from_comp) + 1 == critical_path.index(to_comp)) else 'black'
                dot.edge(from_comp, to_comp, color=edge_color)
        
        dot.render(filename, format='png', cleanup=True)
        print(f"Граф с критическим путём сохранен как {filename}.png")

def parse_input_file(filename: str) -> DependencyGraph:
    """Парсит входной файл с зависимостями"""
    graph = DependencyGraph()
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if 'зависит от' in line:
                parts = line.split('зависит от')
                component = parts[0].strip()
                dependencies = [dep.strip() for dep in parts[1].split(',')]
                
                graph.add_component(component)
                for dep in dependencies:
                    if dep:  # проверка на пустую строку
                        graph.add_dependency(component, dep)
    
    return graph

# Пример использования
def main():
    # Создаем граф зависимостей
    graph = DependencyGraph()
    
    # Добавляем зависимости из примера
    graph.add_dependency('A', 'B')
    graph.add_dependency('A', 'C')
    graph.add_dependency('B', 'D')
    graph.add_dependency('C', 'D')
    graph.add_dependency('C', 'E')
    graph.add_dependency('E', 'B')
    
    # Проверяем ацикличность и получаем топологический порядок
    try:
        topological_order = graph.get_topological_order()
        print("Топологический порядок:", topological_order)
        print("Граф ацикличный:", graph.is_acyclic())
    except ValueError as e:
        print("Ошибка:", e)
    
    # Анализ зависимостей
    analyzer = DependencyAnalyzer(graph)
    
    # BFS анализ для компонента A
    bfs_result = analyzer.find_dependencies_bfs('A')
    print("BFS зависимости для A:", bfs_result)
    
    # DFS анализ для компонента A
    dfs_result = analyzer.find_dependencies_dfs('A')
    print("DFS зависимости для A:", dfs_result)
    
    # Продвинутый анализ с весами
    advanced_analyzer = AdvancedDependencyAnalyzer(graph)
    
    # Добавляем веса рёбрам (например, время сборки)
    advanced_analyzer.add_weight('A', 'B', 2.0)
    advanced_analyzer.add_weight('A', 'C', 1.5)
    advanced_analyzer.add_weight('B', 'D', 3.0)
    advanced_analyzer.add_weight('C', 'D', 2.0)
    advanced_analyzer.add_weight('C', 'E', 1.0)
    advanced_analyzer.add_weight('E', 'B', 0.5)
    
    # Находим критический путь
    critical_path, total_time = advanced_analyzer.find_critical_path('A')
    print("Критический путь для A:", critical_path)
    print("Общее время критического пути:", total_time)
    
    # Визуализация
    GraphVisualizer.visualize(graph, "basic_dependency_graph")
    GraphVisualizer.visualize_with_critical_path(graph, critical_path, "critical_path_graph")

if __name__ == "__main__":
    main()
