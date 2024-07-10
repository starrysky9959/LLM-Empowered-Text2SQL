import networkx as nx
import sqlalchemy


class ForeignKeyGraph:
    def __init__(self, engine):
        self.graph = self.build_graph(engine)

    def build_graph(self, engine):
        inspector = sqlalchemy.inspect(engine)
        table_names = inspector.get_table_names()
        G = nx.DiGraph()
        for table_name in table_names:
            G.add_node(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            for foreign_key in foreign_keys:
                ref_table = foreign_key["referred_table"]
                G.add_edge(table_name, ref_table)

                # WARNNING!!!!!
                # ill-formed foreign key name in bird dataset
                if ref_table not in table_names:
                    for tmp_table_name in table_names:
                        if tmp_table_name.lower() == ref_table.lower():
                            ref_table = tmp_table_name
                            break
                        
                G.add_edge(ref_table, table_name)
                # print("fk")
                # print(table_name)
                # print(ref_table)
                
                
                
        return G

    def one_hop(self, table_names):
        results = set(table_names)
        for table_name in table_names:
            if table_name in self.graph:
                neighbors = set(self.graph.neighbors(table_name))
                results.update(neighbors)
            else:
                assert False
        return results

    def two_hop(self, table_names):
        results = {}
        for table_name in table_names:
            if table_name in self.graph:
                neighbors = list(self.graph.neighbors(table_name))
                two_hop_neighbors = set(neighbors)
                for neighbor in neighbors:
                    two_hop_neighbors.update(self.graph.neighbors(neighbor))
                # 移除直接邻居和自身
                two_hop_neighbors.discard(table_name)
                # 将本身和直接邻居插入到结果的最前面
                result = [table_name] + neighbors + list(two_hop_neighbors)
                results[table_name] = result
            else:
                results[table_name] = [table_name] if table_name in self.graph else []
        return results
