from modules import *


def get_model(args):
	model = UnifiedSSR(u_vocab=args.user_vocab,
	                   p_vocab=args.product_vocab,
	                   t_vocab=args.term_vocab,
	                   emb_size=args.emb_size,
	                   hid_size=args.hid_size,
	                   sub_seq_num=args.sub_seq_num,
	                   enc_num_layer=args.enc_num_layer,
	                   num_head=args.num_head,
	                   tasks=args.tasks,
	                   dropout=args.dropout)
	return model


class UnifiedSSR(nn.Module):
	def __init__(self, u_vocab, p_vocab, t_vocab, emb_size, hid_size, sub_seq_num, enc_num_layer,
	             num_head, tasks, padding_value=0, dropout=0.1):
		super(UnifiedSSR, self).__init__()
		hid_size = hid_size or emb_size * 2
		self.tasks = tasks
		self.p_vocab = p_vocab
		self.t_vocab = t_vocab
		self.sub_seq_num = sub_seq_num
		self.emb_size = emb_size
		self.padding_value = padding_value
		self.u_embed = Embeddings(u_vocab, emb_size)
		self.p_embed = Embeddings(p_vocab, emb_size)
		self.position = PositionalEncoding(emb_size, dropout)
		self.q_t_embed = Embeddings(t_vocab + 2, emb_size)  # additional bos, eos
		self.encoder = SiameseEncoder(SiameseEncoderLayer(emb_size, hid_size, num_head, dropout), enc_num_layer)
		self.seq_partition = SequencePartition(sub_seq_num, emb_size)
		self.next_product_search_w = nn.Parameter(torch.tensor(0.5))
		self.loss = None

	def forward(self, task, inputs):
		"""
		Shape:
			(task == 'recommendation' -> next product prediction)
			:return p_enc: [BS, Seq Max Len, Emb Size]

			(task == 'search' -> next product retrieval)
			:return p_enc: [BS, Seq Max Len, Emb Size]
			:return q_enc: [BS, Seq Max Len, Emb Size]
		"""
		if task == 'recommendation':
			self.loss = self.next_product_predict_loss
			p_rep = self.position(self.p_embed(inputs['pids_in']) + self.u_embed(inputs['uid']).unsqueeze(1))
			p_enc = self.encoder(p_rep, p_rep, inputs['pids_mask'])
			return p_enc
		else:  # task == 'search'
			self.loss = self.next_product_search_loss
			p_rep = self.position(self.p_embed(inputs['pids_in']) + self.u_embed(inputs['uid']).unsqueeze(1))
			q_rep = [[torch.mean(self.q_t_embed(qry), dim=0) for qry in qrys] for qrys in inputs['qrys_in']]
			q_rep = torch.stack([torch.stack(q_rep_t) for q_rep_t in q_rep])
			q_rep = self.position(q_rep + self.u_embed(inputs['uid']).unsqueeze(1))
			p_enc = self.encoder(p_rep, q_rep, inputs['pids_mask'])
			q_enc = self.encoder(q_rep, p_rep, inputs['qrys_in_mask'])
			return p_enc, q_enc

	def next_product_predict_loss(self, seq_emb, mask, p_pos, p_negs):
		p_pos_emb = self.p_embed(p_pos)
		# p_pos_emb [BS*MaxLen, EmbSize]
		p_pos_logits = torch.sum(p_pos_emb * seq_emb, -1)
		# p_pos_logits [BS*MaxLen]

		p_negs_emb = self.p_embed(p_negs)
		# p_negs_emb [BS*MaxLen, NumNeg, EmbSize]
		p_negs_logits = torch.sum(p_negs_emb * seq_emb.unsqueeze(1).repeat(1, p_negs_emb.size(1), 1), -1)
		# p_negs_logits [BS*MaxLen, NumNeg]

		loss = - torch.sum(
			torch.log(p_pos_logits.sigmoid() + 1e-24) * mask +
			torch.log(1 - p_negs_logits.sigmoid() + 1e-24).sum(-1) * mask
		) / mask.sum()

		return loss

	def next_product_predict(self, seq_emb, last_idx, p_pred=None):
		last_idx = last_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, seq_emb.size(-1))
		seq_last_out = seq_emb.gather(1, last_idx).squeeze()
		if p_pred is not None:
			p_emb = self.p_embed(p_pred)
			# p_emb [BS, NumNeg+1, EmbSize]
			seq_last_out = seq_last_out.unsqueeze(1).repeat(1, p_emb.size(1), 1)
			# seq_last_out [BS, NumNeg+1 or PVocab, EmbSize]
			pred_logits = torch.sum(p_emb * seq_last_out, -1)
			# pred_logits [BS, NumNeg+1 or PVocab]
			return pred_logits
		else:
			p_emb = self.p_embed.lut.weight * math.sqrt(self.emb_size)
			# p_emb = self.p_embed.lut.weight
			# p_emb [PVocab, EmbSize]
			p_emb = p_emb.unsqueeze(0).repeat(seq_emb.size(0), 1, 1)
			# p_emb [BS, PVocab, EmbSize]
			p_emb_chunks = [p_emb[:, i:i + 5000] for i in range(0, p_emb.size(1), 5000)]
			pred_logits = []
			for p_emb_chunk in p_emb_chunks:
				if p_emb.device.type == 'cuda':
					pred_logits.append(
						torch.sum(p_emb_chunk * seq_last_out.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1).cpu())
				else:
					pred_logits.append(
						torch.sum(p_emb_chunk * seq_last_out.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1))
			return torch.cat(pred_logits, dim=1)

	def next_product_search_loss(self, p_seq_emb, q_seq_emb, mask, p_pos, p_negs):
		p_pos_emb = self.p_embed(p_pos)
		# p_pos_emb [BS*MaxLen, EmbSize]
		p_negs_emb = self.p_embed(p_negs)
		# p_negs_emb [BS*MaxLen, NumNeg, EmbSize]

		p_pos_sc = torch.sum(p_pos_emb * p_seq_emb, -1)
		# p_pos_sc [BS*MaxLen]
		p_negs_sc = torch.sum(p_negs_emb * p_seq_emb.unsqueeze(1).repeat(1, p_negs_emb.size(1), 1), -1)
		# p_negs_sc [BS*MaxLen, NumNeg]
		p_loss = - torch.sum(
			torch.log(p_pos_sc.sigmoid() + 1e-24) * mask +
			torch.log(1 - p_negs_sc.sigmoid() + 1e-24).sum(-1) * mask
		) / mask.sum()

		q_pos_sc = torch.sum(p_pos_emb * q_seq_emb, -1)
		# q_pos_sc [BS*MaxLen]
		q_negs_sc = torch.sum(p_negs_emb * q_seq_emb.unsqueeze(1).repeat(1, p_negs_emb.size(1), 1), -1)
		# q_negs_sc [BS*MaxLen,NumNeg]
		q_loss = - torch.sum(
			torch.log(q_pos_sc.sigmoid() + 1e-24) * mask +
			torch.log(1 - q_negs_sc.sigmoid() + 1e-24).sum(-1) * mask
		) / mask.sum()

		self.next_product_search_w.data = self.next_product_search_w.clamp(min=0.1, max=0.9)
		return self.next_product_search_w * p_loss + (1 - self.next_product_search_w) * q_loss

	def next_product_search(self, p_seq_emb, q_seq_emb, last_idx, p_pred=None):
		last_idx = last_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, p_seq_emb.size(-1))
		p_seq_last_out = p_seq_emb.gather(1, last_idx).squeeze()  # [BS, EmbSize]
		q_seq_last_out = q_seq_emb.gather(1, last_idx).squeeze()  # [BS, EmbSize]

		if p_pred is not None:
			p_emb = self.p_embed(p_pred)
			# p_emb [BS, NumNeg+1, EmbSize]

			p_seq_last_out = p_seq_last_out.unsqueeze(1).repeat(1, p_emb.size(1), 1)
			# p_seq_last_out [BS, NumNeg+1, EmbSize]
			q_seq_last_out = q_seq_last_out.unsqueeze(1).repeat(1, p_emb.size(1), 1)
			# q_seq_last_out [BS, NumNeg+1, EmbSize]
			p_pred_logits = torch.sum(p_emb * p_seq_last_out, -1)  # [BS, NumNeg+1]
			q_pred_logits = torch.sum(p_emb * q_seq_last_out, -1)  # [BS, NumNeg+1]
			return self.next_product_search_w * p_pred_logits + (1 - self.next_product_search_w) * q_pred_logits
		else:
			p_emb = self.p_embed.lut.weight * math.sqrt(self.emb_size)  # [PVocab, EmbSize]
			# p_emb = self.p_embed.lut.weight # [PVocab, EmbSize]
			p_emb = p_emb.unsqueeze(0).repeat(p_seq_emb.size(0), 1, 1)
			# p_emb [BS, PVocab, EmbSize]
			p_emb_chunks = [p_emb[:, i:i + 5000] for i in range(0, p_emb.size(1), 5000)]
			pred_logits = []
			for p_emb_chunk in p_emb_chunks:
				p_pred_logits_ = torch.sum(
					p_emb_chunk * p_seq_last_out.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1)
				q_pred_logits_ = torch.sum(
					p_emb_chunk * q_seq_last_out.unsqueeze(1).repeat(1, p_emb_chunk.size(1), 1), -1)
				pred_logits_ = self.next_product_search_w * p_pred_logits_ + (1 - self.next_product_search_w) * q_pred_logits_
				if p_emb.device.type == 'cuda':
					pred_logits.append(pred_logits_.cpu())
				else:
					pred_logits.append(pred_logits_)
			return torch.cat(pred_logits, dim=1)

	def get_sub_seq_wins(self, emb):
		sub_seq_wins = self.seq_partition(emb)  # [BS, Sub Seq Num, 2]
		return sub_seq_wins

	def intra_corr_loss(self, emb, sub_seq_wins, mask):
		len_idx = torch.arange(emb.size(1), device=emb.device).unsqueeze(0)  # [1, Seq Max Len]
		sub_mask = (sub_seq_wins[:, :, 0:1] <= len_idx) & (len_idx <= sub_seq_wins[:, :, 1:2])
		sub_mask = sub_mask & mask
		# sub_mask [BS, Sub Seq Num, Seq Max Len]
		sub_mask = sub_mask.unsqueeze(-1).expand(-1, -1, -1, emb.size(-1))
		# sub_mask [BS, Sub Seq Num, Seq Max Len, Emb Size]
		sub_seq_rep = emb.unsqueeze(1) * sub_mask.float()
		emb = emb + sub_seq_rep.sum(dim=1) / (sub_mask.sum(dim=1) + 1e-10)

		sub_seq_rep = sub_seq_rep.sum(dim=-2) / (sub_mask.sum(dim=-2) + 1e-10)
		intra_corr = F.cosine_similarity(sub_seq_rep.unsqueeze(2), sub_seq_rep.unsqueeze(1), dim=-1)
		intra_corr = torch.abs(intra_corr)
		corr_mask = torch.triu(torch.ones((1, sub_seq_rep.size(1), sub_seq_rep.size(1)), device=sub_seq_rep.device),
		                       diagonal=1).bool()
		corr_mask = corr_mask & ~torch.triu(
			torch.ones((1, sub_seq_rep.size(1), sub_seq_rep.size(1)), device=sub_seq_rep.device), diagonal=2).bool()
		intra_corr = intra_corr * corr_mask.float()
		intra_corr_loss = intra_corr.sum() / (intra_corr.nonzero().size(0) + 1e-10)
		return emb, sub_seq_rep, intra_corr_loss

	def inter_corr_loss(self, p_emb, p_sub_seq_wins, q_emb, q_sub_seq_wins, mask):
		p_emb, p_sub_seq_rep, p_intra_corr_loss = self.intra_corr_loss(p_emb, p_sub_seq_wins, mask)
		q_emb, q_sub_seq_rep, q_intra_corr_loss = self.intra_corr_loss(q_emb, q_sub_seq_wins, mask)
		inter_corr = F.cosine_similarity(p_sub_seq_rep, q_sub_seq_rep, dim=-1)
		inter_corr_loss = inter_corr.sum() / (inter_corr.nonzero().size(0) + 1e-10)
		inter_corr_loss = p_intra_corr_loss + q_intra_corr_loss - inter_corr_loss
		return p_emb, q_emb, inter_corr_loss
