import torch
import torch.nn as nn
from base.base_model import BaseModel

class Recommender(BaseModel):

    def __init__(self, config, doc_feature_embedding, entity_embedding):
        super(Recommender, self).__init__()
        self.config = config
        self.doc_feature_embedding = doc_feature_embedding
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)

        self.elu = nn.ELU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.mlp_layer1 = nn.Linear(self.config['model']['embedding_size']*2, self.config['model']['embedding_size'])
        self.mlp_layer2 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.news_compress1 = nn.Linear(self.config['model']['doc_embedding_size'], self.config['model']['embedding_size'])
        self.news_compress2 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.doc_feature_embedding[newsid]).cuda())
        return torch.stack(news_embeddings)

    def get_anchor_graph_embedding(self, anchor_graph_list):
        anchor_graph_embedding_list = []
        for i in range(len(anchor_graph_list)):
            anchor_graph_embedding_list.append(torch.sum(self.entity_embedding(anchor_graph_list[i]), dim=1))
        return torch.stack(anchor_graph_embedding_list)

    def forward(self, news1, news2, anchor_graph1, anchor_graph2):
        news_embedding1 = self.get_news_embedding_batch(news1)
        news_embedding2 = self.get_news_embedding_batch(news2)
        news_embedding1 = self.tanh(self.news_compress2(self.elu(self.news_compress1(news_embedding1))))
        news_embedding2 = self.tanh(self.news_compress2(self.elu(self.news_compress1(news_embedding2))))
        anchor_embedding1 = self.get_anchor_graph_embedding(anchor_graph1)
        anchor_embedding2 = self.get_anchor_graph_embedding(anchor_graph2)

        anchor_embedding1 = torch.sum(anchor_embedding1, dim=0)
        anchor_embedding2 = torch.sum(anchor_embedding2, dim=0)
        news_embedding1 = torch.cat([news_embedding1,anchor_embedding1], dim=-1)
        news_embedding2 = torch.cat([news_embedding2,anchor_embedding2], dim=-1)

        news_embedding1 = self.elu(self.mlp_layer2(self.elu(self.mlp_layer1(news_embedding1))))
        news_embedding2 = self.elu(self.mlp_layer2(self.elu(self.mlp_layer1(news_embedding2))))
        predict = (self.cos(news_embedding1, news_embedding2)+1)/2
        return predict, news_embedding1