import pandas as pd


class HierarchyTree:
    def __init__(self, path):
        self.hierarchy = pd.read_csv(path)

        self.tree = [
            {
                "class": "root",
                "parent": None,
                "children": [],
                "level": "N0",
            }
        ]

        self.levels = ['N0', 'N1', 'N2', 'N3']

        self.mount()


    def get_node(self, categ, level):
        for node in self.tree:
            if node['class'] == categ and node['level'] == level:
                return node

        return None


    def remove_node(self, categ, level):
        node = self.get_node(categ, level)
        self.tree.remove(node)

        level = self.levels.index(node['level']) - 1
        
        parent = self.get_node(node['parent'], self.levels[level])
        parent['children'].remove(categ)


    def mount(self):
        for categ in self.hierarchy['N1'].unique():
            # Adiciona o categ ao children do pai
            self.tree[0]['children'].append(categ)

            # Cria o nó
            node = {
                "class": categ,
                "parent": self.tree[0]["class"],
                "children": [],
                "level": "N1"
            }

            # Adiciona a árvore
            self.tree.append(node)

        for categ in self.hierarchy['N2'].unique():
            # Adiciona o categ ao children do pai
            parent = self.hierarchy[self.hierarchy['N2'] == categ]['N1'].iloc[0]
            parent_index = next((index for (index, d) in enumerate(self.tree) if d["class"] == parent), None)
            self.tree[parent_index]['children'].append(categ)

            # Cria o nó
            node = {
                "class": categ,
                "parent": self.tree[parent_index]["class"],
                "children": [],
                "level": "N2"
            }

            # Adiciona a árvore
            self.tree.append(node)

        for categ in self.hierarchy['N3'].unique():
            # Adiciona o categ ao children do pai
            parent = self.hierarchy[self.hierarchy['N3'] == categ]['N2'].iloc[0]
            parent_index = next((index for (index, d) in enumerate(self.tree) if d["class"] == parent), None)
            self.tree[parent_index]['children'].append(categ)

            # Cria o nó
            node = {
                "class": categ,
                "parent": self.tree[parent_index]["class"],
                "children": [],
                "level": "N3"
            }

            # Adiciona a árvore
            self.tree.append(node)

        # for node in [
        #     "Estelionato (outros) – Tentativa",
        #     "Roubo de Veículo – Moto",
        #     "Furto de Veículo – Moto",
        #     "Homicídio Provocado por Projétil de Arma de Fogo – Tentativa"]:
        #     self.remove_node(node, "N3")


    def get_level_nodes(self, level):
        nodes = []
        for node in self.tree:
            if node['level'] == level:
                nodes.append(node)

        return nodes


    def print(self):
        print(f"{self.tree[0]['class']}")
        for child in self.tree[0]['children']:
            node = self.get_node(child, 'N1')
            print(f"\t|- {node['class']}")
            for child_2 in node['children']:
                node_2 = self.get_node(child_2, 'N2')
                print(f"\t|\t|- {node_2['class']}")
                for child_3 in node_2['children']:
                    node_3 = self.get_node(child_3, 'N3')
                    print(f"\t|\t|\t|- {node_3['class']}")


    def get_levels(self):
        return self.levels
