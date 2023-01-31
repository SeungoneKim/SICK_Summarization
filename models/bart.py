import transformers
from transformers.models.bart.modeling_bart import *
import sys
sys.path.append('../')
from utils.util import load_checkpoint

#################################################################################################################################
class BartModel_DualDecoder(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.extra_decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize extra_decoder_weights with original decoder_weights
        self.extra_decoder = load_checkpoint(self.decoder, self.extra_decoder)
        print('Extra matching is done!')

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
        self.extra_decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_extra_decoder(self):
        return self.extra_decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_extra_input_ids=None,
        decoder_attention_mask=None,
        decoder_extra_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        
        # Training only
        if self.training:
            if decoder_extra_input_ids is None and decoder_inputs_embeds is None:
                if input_ids is None:
                    raise ValueError(
                        "If no `decoder_extra_input_ids` or `decoder_extra_inputs_embeds` are "
                        "passed, `extra_input_ids` cannot be `None`. Please pass either "
                        "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                    )

                decoder_extra_input_ids = shift_tokens_right(
                    input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        if self.training:
            extra_decoder_outputs = self.extra_decoder(
                input_ids=decoder_extra_input_ids,
                attention_mask=decoder_extra_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.training:
            return (Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,),
                Seq2SeqModelOutput(
                last_hidden_state=extra_decoder_outputs.last_hidden_state,
                past_key_values=extra_decoder_outputs.past_key_values,
                decoder_hidden_states=extra_decoder_outputs.hidden_states,
                decoder_attentions=extra_decoder_outputs.attentions,
                cross_attentions=extra_decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,))
        else:
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,)

#################################################################################################################################


#################################################################################################################################
class BartForConditionalGeneration_DualDecoder(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel_DualDecoder(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("extra_final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.extra_lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_extra_decoder(self):
        return self.model.get_extra_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.extra_lm_head = super()._get_resized_lm_head(self.extra_lm_head, new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
            new_bias2 = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
            extra_bias2 = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.extra_final_logits_bias.device)
            new_bias2 = torch.cat([self.extra_final_logits_bias, extra_bias2], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
        self.register_buffer("extra_final_logits_bias", new_bias2)

    def get_output_embeddings(self):
        return self.lm_head
    
    def get_extra_output_embeddings(self):
        return self.extra_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.extra_lm_head = new_embeddings


    def forward(
        self,
        input_ids=None, # x or x||z
        attention_mask=None,
        decoder_input_ids=None,
        decoder_extra_input_ids=None,
        decoder_attention_mask=None,
        decoder_extra_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        extra_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Train, Validation
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        # Train
        if extra_labels is not None:
            if decoder_extra_input_ids is None and decoder_inputs_embeds is None:
                decoder_extra_input_ids = shift_tokens_right(
                    extra_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        # Train
        if extra_labels is not None:
            outputs, extra_outputs = self.model(
                input_ids, # x or x||z
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, # y_shift
                decoder_extra_input_ids=decoder_extra_input_ids, # w_shift
                encoder_outputs=encoder_outputs, # E(x||z)
                decoder_attention_mask=decoder_attention_mask,
                decoder_extra_attention_mask=decoder_extra_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Validation, Test
        else:
            outputs = self.model(
                input_ids, # x or x||z
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, # y_shift
                decoder_extra_input_ids=decoder_extra_input_ids, # w_shift
                encoder_outputs=encoder_outputs, # E(x||z)
                decoder_attention_mask=decoder_attention_mask,
                decoder_extra_attention_mask=decoder_extra_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # Train
        if extra_labels is not None:
            extra_lm_logits = self.extra_lm_head(extra_outputs[0]) + self.extra_final_logits_bias

        masked_lm_loss1 = None
        masked_lm_loss2 = None

        # Train, Validation
        if labels is not None:
            loss_fct1 = CrossEntropyLoss()
            masked_lm_loss1 = loss_fct1(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        # Train
        if extra_labels is not None:
            loss_fct2 = CrossEntropyLoss()
            masked_lm_loss2 = loss_fct2(extra_lm_logits.view(-1, self.config.vocab_size), extra_labels.view(-1))


        if not return_dict:
            output1 = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss1,) + output1) if masked_lm_loss1 is not None else output
        
        # Train
        if extra_labels is not None:
            return (Seq2SeqLMOutput(
                loss=masked_lm_loss1,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            ),Seq2SeqLMOutput(
                loss=masked_lm_loss2,
                logits=extra_lm_logits,
                past_key_values=extra_outputs.past_key_values,
                decoder_hidden_states=extra_outputs.decoder_hidden_states,
                decoder_attentions=extra_outputs.decoder_attentions,
                cross_attentions=extra_outputs.cross_attentions,
                encoder_last_hidden_state=extra_outputs.encoder_last_hidden_state,
                encoder_hidden_states=extra_outputs.encoder_hidden_states,
                encoder_attentions=extra_outputs.encoder_attentions,
            ))
        # Validation, Test
        else:
            #outputs = outputs[0]
            return Seq2SeqLMOutput(
                loss=masked_lm_loss1,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


#################################################################################################################################
class BartForConditionalGeneration_DualHead(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("extra_final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.extra_lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.extra_lm_head = super()._get_resized_lm_head(self.extra_lm_head, new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
            new_bias2 = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
            extra_bias2 = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.extra_final_logits_bias.device)
            new_bias2 = torch.cat([self.extra_final_logits_bias, extra_bias2], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
        self.register_buffer("extra_final_logits_bias", new_bias2)

    def get_output_embeddings(self):
        return self.lm_head
    
    def get_extra_output_embeddings(self):
        return self.extra_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.extra_lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_extra_input_ids=None,
        decoder_attention_mask=None,
        decoder_extra_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        extra_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Train, Validation
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        # Train
        if extra_labels is not None:
            if decoder_extra_input_ids is None and decoder_inputs_embeds is None:
                decoder_extra_input_ids = shift_tokens_right(
                    extra_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # Train
        if extra_labels is not None:
            extra_lm_logits = self.extra_lm_head(outputs[0]) + self.extra_final_logits_bias


        masked_lm_loss1 = None
        masked_lm_loss2 = None

        # Train, Validation
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss1 = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Train
        if extra_labels is not None:
            loss_fct2 = CrossEntropyLoss()
            masked_lm_loss2 = loss_fct2(extra_lm_logits.view(-1, self.config.vocab_size), extra_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss1,) + output) if masked_lm_loss1 is not None else output
        
        # Train
        if extra_labels is not None:
            return (Seq2SeqLMOutput(
                loss=masked_lm_loss1,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            ),Seq2SeqLMOutput(
                loss=masked_lm_loss2,
                logits=extra_lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            ))
        # Validation, Test
        else:
            return Seq2SeqLMOutput(
                loss=masked_lm_loss1,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
###########################################################################################################


class BartForConditionalGeneration_DualHead_viz(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("extra_final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.extra_lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.extra_lm_head = super()._get_resized_lm_head(self.extra_lm_head, new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
            new_bias2 = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
            extra_bias2 = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.extra_final_logits_bias.device)
            new_bias2 = torch.cat([self.extra_final_logits_bias, extra_bias2], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
        self.register_buffer("extra_final_logits_bias", new_bias2)

    def get_output_embeddings(self):
        return self.lm_head
    
    def get_extra_output_embeddings(self):
        return self.extra_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.extra_lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_extra_input_ids=None,
        decoder_attention_mask=None,
        decoder_extra_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        extra_labels=None,
        use_cache=None,
        output_attentions=True,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Train, Validation
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        # Train
        if extra_labels is not None:
            if decoder_extra_input_ids is None and decoder_inputs_embeds is None:
                decoder_extra_input_ids = shift_tokens_right(
                    extra_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # Train
        if extra_labels is not None:
            extra_lm_logits = self.extra_lm_head(outputs[0]) + self.extra_final_logits_bias


        masked_lm_loss1 = None
        masked_lm_loss2 = None

        # Train, Validation
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss1 = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Train
        if extra_labels is not None:
            loss_fct2 = CrossEntropyLoss()
            masked_lm_loss2 = loss_fct2(extra_lm_logits.view(-1, self.config.vocab_size), extra_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss1,) + output) if masked_lm_loss1 is not None else output
        
        # Train
        if extra_labels is not None:
            return (Seq2SeqLMOutput(
                loss=masked_lm_loss1,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            ),Seq2SeqLMOutput(
                loss=masked_lm_loss2,
                logits=extra_lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            ))
        # Validation, Test
        else:
            return Seq2SeqLMOutput(
                loss=masked_lm_loss1,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

class BartForConditionalGeneration_DualDecoder_viz(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel_DualDecoder(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.register_buffer("extra_final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.extra_lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_extra_decoder(self):
        return self.model.get_extra_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.extra_lm_head = super()._get_resized_lm_head(self.extra_lm_head, new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
            new_bias2 = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
            extra_bias2 = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.extra_final_logits_bias.device)
            new_bias2 = torch.cat([self.extra_final_logits_bias, extra_bias2], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
        self.register_buffer("extra_final_logits_bias", new_bias2)

    def get_output_embeddings(self):
        return self.lm_head
    
    def get_extra_output_embeddings(self):
        return self.extra_lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.extra_lm_head = new_embeddings


    def forward(
        self,
        input_ids=None, # x or x||z
        attention_mask=None,
        decoder_input_ids=None,
        decoder_extra_input_ids=None,
        decoder_attention_mask=None,
        decoder_extra_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        extra_labels=None,
        use_cache=None,
        output_attentions=True,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Train, Validation
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        # Train
        if extra_labels is not None:
            if decoder_extra_input_ids is None and decoder_inputs_embeds is None:
                decoder_extra_input_ids = shift_tokens_right(
                    extra_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        # Train
        if extra_labels is not None:
            outputs, extra_outputs = self.model(
                input_ids, # x or x||z
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, # y_shift
                decoder_extra_input_ids=decoder_extra_input_ids, # w_shift
                encoder_outputs=encoder_outputs, # E(x||z)
                decoder_attention_mask=decoder_attention_mask,
                decoder_extra_attention_mask=decoder_extra_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Validation, Test
        else:
            outputs = self.model(
                input_ids, # x or x||z
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, # y_shift
                decoder_extra_input_ids=decoder_extra_input_ids, # w_shift
                encoder_outputs=encoder_outputs, # E(x||z)
                decoder_attention_mask=decoder_attention_mask,
                decoder_extra_attention_mask=decoder_extra_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # Train
        if extra_labels is not None:
            extra_lm_logits = self.extra_lm_head(extra_outputs[0]) + self.extra_final_logits_bias

        masked_lm_loss1 = None
        masked_lm_loss2 = None

        # Train, Validation
        if labels is not None:
            loss_fct1 = CrossEntropyLoss()
            masked_lm_loss1 = loss_fct1(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        # Train
        if extra_labels is not None:
            loss_fct2 = CrossEntropyLoss()
            masked_lm_loss2 = loss_fct2(extra_lm_logits.view(-1, self.config.vocab_size), extra_labels.view(-1))


        if not return_dict:
            output1 = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss1,) + output1) if masked_lm_loss1 is not None else output
        
        # Train
        if extra_labels is not None:
            return (Seq2SeqLMOutput(
                loss=masked_lm_loss1,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            ),Seq2SeqLMOutput(
                loss=masked_lm_loss2,
                logits=extra_lm_logits,
                past_key_values=extra_outputs.past_key_values,
                decoder_hidden_states=extra_outputs.decoder_hidden_states,
                decoder_attentions=extra_outputs.decoder_attentions,
                cross_attentions=extra_outputs.cross_attentions,
                encoder_last_hidden_state=extra_outputs.encoder_last_hidden_state,
                encoder_hidden_states=extra_outputs.encoder_hidden_states,
                encoder_attentions=extra_outputs.encoder_attentions,
            ))
        # Validation, Test
        else:
            return Seq2SeqLMOutput(
                loss=masked_lm_loss1,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
