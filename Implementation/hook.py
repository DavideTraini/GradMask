import torch

class VIT_HOOK:
    def __init__(self, model):
        self.model = model
        
        
        # Inizializza i tensori vuoti
        self.query_gradients = None
        self.key_gradients = None
        self.value_gradients = None
        self.queries = None
        self.keys = None
        self.values = None

        # Aggiungi hook per ogni componente Q, K e V
        for layer in self.model.vit.encoder.layer:
            layer.attention.attention.query.register_full_backward_hook(self.save_query_gradient)
            layer.attention.attention.key.register_full_backward_hook(self.save_key_gradient)
            layer.attention.attention.value.register_full_backward_hook(self.save_value_gradient)

            # layer.attention.attention.query.register_forward_hook(self.save_query)
            # layer.attention.attention.key.register_forward_hook(self.save_key)
            # layer.attention.attention.value.register_forward_hook(self.save_value)

    def save_query(self, module, input, output):
        # Concatenazione dei risultati per creare un singolo tensore per le query
        self.queries = torch.cat((self.queries, output.detach()), dim=0) if self.queries is not None else output.detach()

    def save_key(self, module, input, output):
        # Concatenazione dei risultati per creare un singolo tensore per le key
        self.keys = torch.cat((self.keys, output.detach()), dim=0) if self.keys is not None else output.detach()

    def save_value(self, module, input, output):
        # Concatenazione dei risultati per creare un singolo tensore per le value
        self.values = torch.cat((self.values, output.detach()), dim=0) if self.values is not None else output.detach()

    def save_query_gradient(self, module, grad_input, grad_output):
        # Concatenazione dei gradienti per le query
        self.query_gradients = torch.cat((self.query_gradients, grad_output[0].detach()), dim=0) if self.query_gradients is not None else grad_output[0].detach()

    def save_key_gradient(self, module, grad_input, grad_output):
        # Concatenazione dei gradienti per le key
        self.key_gradients = torch.cat((self.key_gradients, grad_output[0].detach()), dim=0) if self.key_gradients is not None else grad_output[0].detach()

    def save_value_gradient(self, module, grad_input, grad_output):
        # Concatenazione dei gradienti per le value
        self.value_gradients = torch.cat((self.value_gradients, grad_output[0].detach()), dim=0) if self.value_gradients is not None else grad_output[0].detach()

    def clear_hooks(self):
        # Reset dei tensori
        self.query_gradients = None
        self.key_gradients = None
        self.value_gradients = None
        self.queries = None
        self.keys = None
        self.values = None





class DEIT_HOOK:
    def __init__(self, model):
        self.model = model
        
        
        # Inizializza i tensori vuoti
        self.query_gradients = None
        self.key_gradients = None
        self.value_gradients = None
        self.queries = None
        self.keys = None
        self.values = None

        # Aggiungi hook per ogni componente Q, K e V
        for layer in self.model.deit.encoder.layer:
            layer.attention.attention.query.register_full_backward_hook(self.save_query_gradient)
            layer.attention.attention.key.register_full_backward_hook(self.save_key_gradient)
            layer.attention.attention.value.register_full_backward_hook(self.save_value_gradient)

            # layer.attention.attention.query.register_forward_hook(self.save_query)
            # layer.attention.attention.key.register_forward_hook(self.save_key)
            # layer.attention.attention.value.register_forward_hook(self.save_value)

    def save_query(self, module, input, output):
        # Concatenazione dei risultati per creare un singolo tensore per le query
        self.queries = torch.cat((self.queries, output.detach()), dim=0) if self.queries is not None else output.detach()

    def save_key(self, module, input, output):
        # Concatenazione dei risultati per creare un singolo tensore per le key
        self.keys = torch.cat((self.keys, output.detach()), dim=0) if self.keys is not None else output.detach()

    def save_value(self, module, input, output):
        # Concatenazione dei risultati per creare un singolo tensore per le value
        self.values = torch.cat((self.values, output.detach()), dim=0) if self.values is not None else output.detach()

    def save_query_gradient(self, module, grad_input, grad_output):
        # Concatenazione dei gradienti per le query
        self.query_gradients = torch.cat((self.query_gradients, grad_output[0].detach()), dim=0) if self.query_gradients is not None else grad_output[0].detach()

    def save_key_gradient(self, module, grad_input, grad_output):
        # Concatenazione dei gradienti per le key
        self.key_gradients = torch.cat((self.key_gradients, grad_output[0].detach()), dim=0) if self.key_gradients is not None else grad_output[0].detach()

    def save_value_gradient(self, module, grad_input, grad_output):
        # Concatenazione dei gradienti per le value
        self.value_gradients = torch.cat((self.value_gradients, grad_output[0].detach()), dim=0) if self.value_gradients is not None else grad_output[0].detach()

    def clear_hooks(self):
        # Reset dei tensori
        self.query_gradients = None
        self.key_gradients = None
        self.value_gradients = None
        self.queries = None
        self.keys = None
        self.values = None