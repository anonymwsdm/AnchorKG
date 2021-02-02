from base.base_model import BaseModel
import torch
import torch.nn as nn
import networkx as nx
import time

class Net(BaseModel):
    def __init__(self, config, entityid_dict, doc_feature_embedding, entity_embedding):
        super(Net, self).__init__()

        self.config = config
        self.doc_feature_embedding = doc_feature_embedding
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.entityid_dict = entityid_dict

        self.actor_l1 = nn.Linear(self.config['model']['embedding_size']*2, self.config['model']['embedding_size'])
        self.actor_l2 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.actor_l3 = nn.Linear(self.config['model']['embedding_size'],1)

        self.critic_l1 = nn.Linear(self.config['model']['embedding_size']*2, self.config['model']['embedding_size'])
        self.critic_l2 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.critic_l3 = nn.Linear(self.config['model']['embedding_size'], 1)

        self.elu = torch.nn.ELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.doc_feature_embedding[newsid]).cuda())
        return torch.stack(news_embeddings)

    def forward(self, state_input, action_input):
        if len(state_input.shape) < len(action_input.shape):
            if len(action_input.shape) == 3:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1],
                                                 state_input.shape[2])
            else:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], action_input.shape[2],
                                                 state_input.shape[3])

        # Actor
        actor_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        actor_out = self.elu(self.actor_l2(actor_x))
        act_probs = self.softmax(self.actor_l3(actor_out))

        # Critic
        critic_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = self.elu(self.critic_l2(critic_x))
        q_actions = self.critic_l3(critic_out)

        return act_probs, q_actions

class AnchorKG(BaseModel):

    def __init__(self, config, doc_entity_dict, entity_doc_dict, doc_feature_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, kg_env_all, hit_dict, entity_id_dict, neibor_embedding, neibor_num):
        super(AnchorKG, self).__init__()
        self.config = config
        self.doc_entity_dict = doc_entity_dict
        self.entity_doc_dict = entity_doc_dict
        self.doc_feature_embedding = doc_feature_embedding
        self.kg_env_all = kg_env_all
        self.hit_dict = hit_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.MAX_DEPTH = 3
        self.entity_id_dict = entity_id_dict
        self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        self.neibor_num = neibor_num

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()

        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.news_compress_1 = nn.Linear(self.config['model']['doc_embedding_size'],self.config['model']['embedding_size'])
        self.news_compress_2 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.innews_relation = nn.Embedding(1,self.config['model']['embedding_size'])

        self.policy_net = Net(self.config, self.entity_id_dict, self.doc_feature_embedding, entity_embedding)
        self.target_net = Net(self.config, self.entity_id_dict, self.doc_feature_embedding, entity_embedding)

    def get_neiborhood_news_embedding_batch(self, news_embedding, entityids):
        neibor_news_embedding_avg = self.neibor_embedding(entityids)
        neibor_num = []
        for i in range(len(entityids)):
            neibor_num.append(torch.index_select(self.neibor_num, 0, entityids[i]))
        neibor_num = torch.stack(neibor_num)
        if len(neibor_news_embedding_avg.shape) > len(news_embedding.shape):
            news_embedding = torch.unsqueeze(news_embedding, 1)
            neibor_num = torch.unsqueeze(neibor_num, 2)
            news_embedding = news_embedding.expand(news_embedding.shape[0] ,neibor_news_embedding_avg.shape[1] ,news_embedding.shape[2])
            neibor_num = neibor_num.expand(neibor_num.shape[0], neibor_num.shape[1],news_embedding.shape[2])
        neibor_news_embedding_avg = torch.div((neibor_news_embedding_avg - news_embedding), neibor_num)
        return neibor_news_embedding_avg

    def get_sim_reward_batch(self, news_embedding_batch, neibor_news_embedding_avg_batch):
        if len(neibor_news_embedding_avg_batch.shape) > len(news_embedding_batch.shape):
            news_embedding_batch = torch.unsqueeze(news_embedding_batch, 1)
            news_embedding_batch = news_embedding_batch.expand(news_embedding_batch.shape[0] ,neibor_news_embedding_avg_batch.shape[1] ,news_embedding_batch.shape[2])
        cos_rewards = self.cos(news_embedding_batch, neibor_news_embedding_avg_batch)
        return cos_rewards

    def get_hit_rewards_batch(self, newsid_batch, state_id_input_batch):
        hit_rewards = []
        hit_rewards_weak = []
        for i in range(len(newsid_batch)):
            hit_rewards.append([])
            hit_rewards_weak.append([])
            for j in range(len(state_id_input_batch[i])):
                if int(state_id_input_batch[i][j].data.cpu().numpy()) in self.entity_doc_dict and newsid_batch[i] in self.hit_dict:
                    entity_neibor = set(self.entity_doc_dict[int(state_id_input_batch[i][j].data.cpu().numpy())]).discard(newsid_batch[i])
                    news_hit_neibor = self.hit_dict[newsid_batch[i]]
                    if entity_neibor != None:
                        if len(entity_neibor & news_hit_neibor)>0:
                            hit_rewards[-1].append(1.0)
                        else:
                            hit_rewards[-1].append(0.0)
                    else:
                        hit_rewards[-1].append(0.0)
                else:
                    hit_rewards[-1].append(0.0)

        return torch.FloatTensor(hit_rewards).cuda()

    def get_batch_rewards_step(self, hit_reward, sim_reward):
        reward = 0.5*hit_reward + 0.5*sim_reward
        return reward

    def get_reward(self, newsid, news_embedding, state_input):
        entity_value = state_input
        neibor_news_embedding_avg = self.get_neiborhood_news_embedding_batch(news_embedding, entity_value)
        sim_reward = self.get_sim_reward_batch(news_embedding, neibor_news_embedding_avg)
        hit_reward = self.get_hit_rewards_batch(newsid, entity_value)
        reward = self.get_batch_rewards_step(hit_reward, sim_reward)
        return reward

    def get_news_entities_batch(self, newsids):
        news_entities = []
        news_relations = []
        for i in range(len(newsids)):
            news_entities.append(self.doc_entity_dict[newsids[i]])
            news_relations.append([0 for k in range(self.config['model']['news_entity_num'])])
        news_entities = torch.tensor(news_entities).cuda()
        news_relations = torch.tensor(news_relations).cuda()
        return news_entities, news_relations

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.doc_feature_embedding[newsid]).cuda())
        return torch.stack(news_embeddings)

    def get_next_action(self, state_id_input_batch):
        next_action_id = []
        next_action_r_id = []
        for i in range(len(state_id_input_batch)):
            next_action_id.append([])
            next_action_r_id.append([])
            for j in range(len(state_id_input_batch[i])):
                if int(state_id_input_batch[i][j].data.cpu().numpy()) in self.entity_adj:
                    next_action_id[-1].append(self.entity_adj[int(state_id_input_batch[i][j].data.cpu().numpy())])
                    next_action_r_id[-1].append(self.relation_adj[int(state_id_input_batch[i][j].data.cpu().numpy())])
                else:
                    next_action_id[-1].append([0 for k in range(20)])
                    next_action_r_id[-1].append([0 for k in range(20)])
        next_action_space_id = torch.LongTensor(next_action_id).cuda()
        next_state_space_embedding = self.entity_embedding(next_action_space_id)
        next_action_r_id = torch.LongTensor(next_action_r_id).cuda()
        next_action_r = self.relation_embedding(next_action_r_id)
        return next_action_space_id, next_state_space_embedding, next_action_r_id, next_action_r

    def get_advantage(self, curr_reward,  q_value):
        advantage = curr_reward - q_value
        return advantage

    def get_actor_loss(self, act_probs_step, advantage):
        actor_loss = act_probs_step * advantage.detach()
        return -actor_loss

    def get_critic_loss(self, advantage):
        critic_loss = torch.pow(advantage, 2)
        return critic_loss

    def step_update(self, act_probs_step, q_values_step, step_reward, embeddding_reward, path_loss,
                    alpha=0.9, beta = 0.09):

        curr_reward = alpha * step_reward + beta * embeddding_reward + (1-alpha-beta) * path_loss
        advantage = self.get_advantage(curr_reward, q_values_step) #curr_reward - q_values_step
        actor_loss = self.get_actor_loss(torch.log(act_probs_step), advantage) # -act_probs_step * advantage #problem2
        critic_loss = self.get_critic_loss(advantage) #advantage.pow(2)

        return critic_loss, actor_loss

    def get_anchor_nodes(self, weights, q_values, action_id_input, relation_id_input, topk):
        if len(weights.shape) <= 3:
            weights =torch.unsqueeze(weights, 1)
            q_values = torch.unsqueeze(q_values, 1)
            action_id_input = torch.unsqueeze(action_id_input, 1)
            relation_id_input = torch.unsqueeze(relation_id_input, 1)

        topk_weights = torch.topk(weights, topk, dim=-2)
        shape0 = topk_weights[1].shape[0]
        shape1 =  topk_weights[1].shape[1]
        topk_index = topk_weights[1].reshape(topk_weights[1].shape[0] * topk_weights[1].shape[1], topk_weights[1].shape[2])
        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2])
        q_values = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])
        weights = weights.gather(1, topk_index)
        q_values = q_values.gather(1, topk_index)
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1],action_id_input.shape[2])
        relation_id_input = relation_id_input.reshape(relation_id_input.shape[0] * relation_id_input.shape[1], relation_id_input.shape[2])
        state_id_input_value = action_id_input.gather(1, topk_index)
        relation_id_selected = relation_id_input.gather(1, topk_index)

        weights = weights.reshape(shape0, shape1*weights.shape[1])
        q_values = q_values.reshape(shape0, shape1 * q_values.shape[1])
        state_id_input_value = state_id_input_value.reshape(shape0, shape1 * state_id_input_value.shape[1])
        relation_id_selected = relation_id_selected.reshape(shape0, shape1 * relation_id_selected.shape[1])

        return weights, q_values, state_id_input_value, relation_id_selected  # anchor_act_probs, anchor_q_values, anchor_nodes, done

    def get_news_embedding_input(self, entity_ids, news_embeddings):
        entity_ids_index = entity_ids._indices()[0]
        news_embedding_batch = news_embeddings(entity_ids_index)
        return news_embedding_batch

    def get_state_input(self, news_embedding, depth, history_entity_1, history_relation_1):
        if depth == 0:
            state_embedding = news_embedding
        else:
            history_entity_embedding = self.entity_embedding(history_entity_1)
            history_relation_embedding = self.relation_embedding(history_relation_1)
            state_embedding = history_relation_embedding + history_entity_embedding
            state_embedding = torch.mean(state_embedding, dim=1, keepdim=False)
            state_embedding = news_embedding + state_embedding

        return state_embedding

    def forward(self, news1, news2):
        #news1
        depth = 0
        history_entity_1 = []
        history_relation_1 = []

        anchor_graph1 = []
        act_probs_steps1 = []
        step_rewards1 = []
        q_values_steps1 = []

        news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(self.get_news_embedding_batch(news1)))))
        news_embedding_origin = self.get_news_embedding_batch(news1)
        input_entities, input_relations = self.get_news_entities_batch(news1)
        action_embedding = self.entity_embedding(input_entities)
        relation_embedding = self.relation_embedding(input_relations)
        action_embedding = action_embedding + relation_embedding
        action_id = input_entities
        relation_id = input_relations
        state_input = self.get_state_input(news_embedding, depth, history_entity_1, history_relation_1)

        while (depth < self.MAX_DEPTH):
            act_probs, q_values = self.policy_net(state_input, action_embedding)
            topk = self.config['model']['topk'][depth]
            anchor_act_probs, anchor_q_values, anchor_nodes, anchor_relations = self.get_anchor_nodes(act_probs, q_values, action_id, relation_id, topk)
            history_entity_1 = anchor_nodes
            history_relation_1 = anchor_relations
            depth = depth + 1
            state_input = self.get_state_input(news_embedding, depth, history_entity_1, history_relation_1)

            act_probs_steps1.append(anchor_act_probs)
            q_values_steps1.append(anchor_q_values)
            actionid_lookup, action_lookup, action_rid_lookup, action_r_lookup = self.get_next_action(anchor_nodes)
            action_id = actionid_lookup
            relation_id = action_rid_lookup
            action_embedding = self.entity_embedding(action_id) + self.relation_embedding(action_rid_lookup)

            # news_embedding_input = self.get_news_embedding_input(state_id_input, news_embedding)
            anchor_graph1.append(anchor_nodes)
            step_reward = self.get_reward(news1, news_embedding_origin, anchor_nodes)
            step_rewards1.append(step_reward)



        # news2
        depth = 0
        history_entity_2 = []
        history_relation_2 = []
        anchor_graph2 = []
        act_probs_steps2 = []
        step_rewards2 = []
        q_values_steps2 = []

        news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(self.get_news_embedding_batch(news2)))))
        news_embedding_origin = self.get_news_embedding_batch(news2)
        input_entities, input_relations = self.get_news_entities_batch(news2)
        action_embedding = self.entity_embedding(input_entities)
        relation_embedding = self.relation_embedding(input_relations)
        action_embedding = action_embedding + relation_embedding
        action_id = input_entities
        relation_id = input_relations

        state_input = self.get_state_input(news_embedding, depth, history_entity_2, history_relation_2)

        while (depth < self.MAX_DEPTH):
            act_probs, q_values = self.policy_net(state_input, action_embedding)
            topk = self.config['model']['topk'][depth]
            anchor_act_probs, anchor_q_values, anchor_nodes, anchor_relations = self.get_anchor_nodes(act_probs,
                                                                                                      q_values,
                                                                                                      action_id,
                                                                                                      relation_id,
                                                                                                      topk)
            history_entity_2 = anchor_nodes
            history_relation_2 = anchor_relations
            depth = depth + 1
            state_input = self.get_state_input(news_embedding, depth, history_entity_2, history_relation_2)

            act_probs_steps2.append(anchor_act_probs)
            q_values_steps2.append(anchor_q_values)
            actionid_lookup, action_lookup, action_rid_lookup, action_r_lookup = self.get_next_action(anchor_nodes)
            action_id = actionid_lookup
            relation_id = action_rid_lookup
            action_embedding = self.entity_embedding(action_id) + self.relation_embedding(action_rid_lookup)

            anchor_graph2.append(anchor_nodes)
            step_reward = self.get_reward(news2, news_embedding_origin, anchor_nodes)
            step_rewards2.append(step_reward)

        return act_probs_steps1, q_values_steps1, act_probs_steps2, q_values_steps2, step_rewards1, step_rewards2, anchor_graph1, anchor_graph2

