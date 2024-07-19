from transformers.trainer import Trainer

class NLPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        assert 'labels' in inputs, 'Why labels not in inputs ?'
        if self.label_smoother is not None and self.model.config.problem_type == 'single_label_classification':
            return super().compute_loss(model, inputs, return_outputs=return_outputs)   
        
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss