import numpy as np
import math
from transformers import ViTForImageClassification, ViTImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision
from torch.nn.functional import interpolate
from hook import VIT_HOOK, DEIT_HOOK
from feature_extractor import Custom_feature_extractor

class GradMask:

    def __init__(self, model, device):

        # Ensure that the specified model is either 'deit' or 'vit'
        assert model == 'deit' or model == 'vit'

        self.device = torch.device(device)

        if model == 'vit':
            # If the model is ViT, initialize the ViTForImageClassification model, Custom_feature_extractor, and VIT_Hook
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            self.image_processor = Custom_feature_extractor(device, model)
            self.vit_hook = VIT_HOOK(self.model)
        else:
            # If the model is DeiT, initialize the DeiTForImageClassificationWithTeacher model, Custom_feature_extractor, and DEIT_Hook
            self.model = DeiTForImageClassificationWithTeacher.from_pretrained(
                'facebook/deit-base-distilled-patch16-224')
            self.image_processor = Custom_feature_extractor(device, model)
            self.vit_hook = DEIT_HOOK(self.model)

        self.model.to(self.device)

    def get_saliency(self, img_path, masks_layer, label=False):

        # Open and preprocess the input image
        image = Image.open(img_path).convert('RGB')

        image_processed, attentions_scores, logits, predicted_label = self.classify(image, self.model, self.image_processor)

        # Determine the ground truth label
        ground_truth_label = predicted_label if not label else torch.tensor([label])
      
        q, k, v = self.get_gradients(self.vit_hook, masks_layer, logits = logits, class_idx = ground_truth_label)

        mask_q, mask_k, mask_v, final_mask = self.get_masks(q, k, v)
        
        batch_images = self.create_masked_images(mask_q, mask_k, mask_v, image_processed)

        heatmap = self.get_heatmap(self.model, batch_images, final_mask, ground_truth_label)

        return heatmap.to('cpu'), ground_truth_label.item()


    def classify(self, image, model, image_processor):
        """
        Classifies an image using the specified model and image processor.

        Args:
            image (torch.Tensor): The input image tensor.
            model (torch.nn.Module): The classification model.
            image_processor (Custom_feature_extractor): The image processor.

        Returns:
            tuple: A tuple containing input features, embedding, and the predicted class index.
        """
        # Process the input image using the provided image processor
        inputs = image_processor(images=image, return_tensors="pt")

        # Forward pass through the model with output_hidden_states
        output = model(**inputs, output_attentions=True)

        logits = output.logits
        attentions_scores = output.attentions

        # Compute softmax probabilities and predict the class index
        probabilities = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1)

        return inputs, attentions_scores, logits, predicted_class_idx

   
    def get_gradients(self, vit_hook, num_masks, logits, class_idx = None, clamp = True):
    
        if class_idx == None:
            class_idx = outputs.logits.argmax(dim=1)
            
        # Definisci la loss rispetto alla classe target
        loss = logits[0, class_idx]
    
        # Calcola il gradiente
        loss.backward()
            
        # Gradiente e valore di ogni componente: query, key, value
        query_gradients = vit_hook.query_gradients
        key_gradients = vit_hook.key_gradients
        value_gradients = vit_hook.value_gradients
    
        vit_hook.clear_hooks()
        
        # Applica la funzione di trasposizione per i gradienti Q, K e V
        query_gradients_transposed = self.transpose_for_scores(query_gradients, num_masks).sum(dim = -1)
        key_gradients_transposed = self.transpose_for_scores(key_gradients, num_masks).sum(dim = -1)
        value_gradients_transposed = self.transpose_for_scores(value_gradients, num_masks).sum(dim = -1)
        
        if clamp:
            query_gradients_transposed = torch.clamp(query_gradients_transposed, min=0.)
            key_gradients_transposed = torch.clamp(key_gradients_transposed, min=0.)
            value_gradients_transposed = torch.clamp(value_gradients_transposed, min=0.)

        if isinstance(self.model, ViTForImageClassification):
          offset = 1
        else:
          offset = 2

        return query_gradients_transposed[1:, :, offset:], key_gradients_transposed[1:, :, offset:], value_gradients_transposed[1:, :, offset:]
    
         


    def get_masks(self, q, k, v):
        L, H, N = q.shape
    
        q_head = q.view(L*H, N)  # (L, N)
        k_head = k.view(L*H, N)  # (L, N)
        v_head = v.view(L*H, N)  # (L, N)
        
        # Calcoliamo le medie lungo la dimensione 1
        mean_q = q_head.mean(dim=1, keepdim=True)  # (L, 1)
        mean_k = k_head.mean(dim=1, keepdim=True)  # (L, 1)
        mean_v = v_head.mean(dim=1, keepdim=True)  # (L, 1)
        
        # Creiamo maschere binarie
        mask_q = (q_head >= mean_q).float()  # (L, 196)
        mask_k = (k_head >= mean_k).float()  # (L, 196)
        mask_v = (v_head >= mean_v).float()  # (L, 196)
        
        # Concatenazione delle maschere per ottenere un tensore finale con 3 * L elementi
        final_mask = torch.cat([mask_q, mask_k, mask_v], dim=0)  # (3 * L, 196)
    
        return mask_q, mask_k, mask_v, final_mask
              

    def create_masked_images(self, mask_q, mask_k, mask_v, inputs):
    
        L = mask_q.shape[0]
    
        # Supponiamo che mask_q, mask_k, mask_v abbiano dimensione (L, 196)
        # Ridimensioniamo le maschere per adattarle a 224x224
        mask_q = mask_q.view(L, 1, 14, 14)  # (L, 1, 14, 14)
        mask_k = mask_k.view(L, 1, 14, 14)  # (L, 1, 14, 14)
        mask_v = mask_v.view(L, 1, 14, 14)  # (L, 1, 14, 14)
        
        # Interpolazione per portare le maschere a 224x224
        mask_q_resized = F.interpolate(mask_q, size=(224, 224), mode='nearest')  # (L, 1, 224, 224)
        mask_k_resized = F.interpolate(mask_k, size=(224, 224), mode='nearest')  # (L, 1, 224, 224)
        mask_v_resized = F.interpolate(mask_v, size=(224, 224), mode='nearest')  # (L, 1, 224, 224)
        
        # Creazione della lista per le immagini oscurate
        masked_images = []
        
        # Applica ogni maschera su inputs (1, C, 224, 224)
        for mask_set in [mask_q_resized, mask_k_resized, mask_v_resized]:
            for i in range(L):
                masked_image = inputs['pixel_values'].clone()  # Copia dell'input originale
                masked_image *= mask_set[i]  # Applica la maschera binaria ridimensionata
                masked_images.append(masked_image)
    
        return torch.cat(masked_images, dim=0)

  
    def modify_image(self, operation, heatmap, image, percentage, baseline, device):
        """
        Modifies an image based on the given operation, heatmap, and baseline.

        Args:
            operation (str): The operation to perform ('deletion' or 'insertion').
            heatmap (torch.Tensor): The heatmap indicating pixel importance.
            image (dict): The image dictionary containing 'pixel_values'.
            percentage (float): The percentage of top pixels to consider for modification.
            baseline (str): The baseline image type ('black', 'blur', 'random', or 'mean').
            device: The device on which to perform the operation.

        Returns:
            torch.Tensor: The modified image tensor.
        """
        if operation not in ['deletion', 'insertion']:
            raise ValueError("Operation must be either 'deletion' or 'insertion'.")

        # Finding the top percentage of most important pixels in the heatmap
        num_top_pixels = int(percentage * heatmap.shape[0] * heatmap.shape[1])
        top_pixels_indices = np.unravel_index(np.argsort(heatmap.ravel())[-num_top_pixels:], heatmap.shape)

        # Extract and copy the image tensor
        img_tensor = image['pixel_values'].squeeze(0)
        img_tensor = img_tensor.permute(1, 2, 0)
        modified_image = np.copy(img_tensor.cpu().numpy())

        tensor_img_reshaped = img_tensor.permute(2, 0, 1)

        # Define baseline image based on the specified type
        if baseline == "black":
            img_baseline = torch.zeros(tensor_img_reshaped.shape, dtype=bool).to(device)
        elif baseline == "blur":
            img_baseline = torchvision.transforms.functional.gaussian_blur(tensor_img_reshaped, kernel_size=[15, 15],
                                                                           sigma=[7, 7])
        elif baseline == "random":
            img_baseline = torch.randn_like(tensor_img_reshaped)
        elif baseline == "mean":
            img_baseline = torch.ones_like(tensor_img_reshaped) * tensor_img_reshaped.mean()

        if operation == 'deletion':
            # Replace the most important pixels by applying the baseline values
            darken_mask = torch.zeros(heatmap.shape, dtype=bool).to(device)
            darken_mask[top_pixels_indices] = 1
            modified_image = torch.where(darken_mask > 0, img_baseline, tensor_img_reshaped)

        elif operation == 'insertion':
            # Replace the less important pixels by applying the baseline values
            keep_mask = torch.zeros(heatmap.shape, dtype=bool).to(device)
            keep_mask[top_pixels_indices] = 1
            modified_image = torch.where(keep_mask > 0, tensor_img_reshaped, img_baseline)

        return modified_image

    
    
    def get_heatmap(self, model, batch_images, final_mask, class_idx):
        with torch.no_grad():
            outputs = model(batch_images)  # Output di dimensione (3 * L, num_classes)
        
        logits = outputs.logits
        
        # Estrai le confidenze per la classe target
        confidences = torch.softmax(logits, dim=1)[:, class_idx]  # (3 * L,)
        
        # Ridimensioniamo le confidenze e moltiplichiamo per la maschera finale
        # Aggiungiamo una dimensione extra per permettere la trasmissione del prodotto
        weighted_masks = final_mask * confidences.view(-1, 1)  # (3 * L, 196)
        
        # Estrai le confidenze per la classe target
        confidences = torch.softmax(logits, dim=1)[:, class_idx]  # (3 * L,)
        
        # Ridimensioniamo le confidenze e moltiplichiamo per la maschera finale
        # Aggiungiamo una dimensione extra per permettere la trasmissione del prodotto
        weighted_masks = final_mask * confidences.view(-1, 1)  # (3 * L, 196)
                
        # Somma i pesi per ogni token lungo la dimensione batch per ottenere una heatmap complessiva
        heatmap_ground_truth_class = torch.sum(weighted_masks, dim=0)  # Somma lungo la dimensione 0, ottieni (196,)
        # coverage_bias = torch.sum(final_mask, dim=0)  # Somma le maschere binarie lungo la dimensione 0
        
        # # Normalizza la heatmap rispetto alla copertura dei token
        # coverage_bias = torch.where(coverage_bias > 0, coverage_bias, 1)  # Evita la divisione per zero
        # heatmap_ground_truth_class = heatmap_ground_truth_class / coverage_bias  # Normalizzazione
        heatmap_ground_truth_class_reshape = heatmap_ground_truth_class.reshape((14, 14))  # Reshape in (14, 14)
        
        return heatmap_ground_truth_class_reshape



  
    # def get_heatmap(self, model, batch_images, final_mask, class_idx):
    #     with torch.no_grad():
    #         outputs = model(batch_images)  # Output di dimensione (3 * L, num_classes)
    
    #     logits = outputs.logits
        
    #     # Estrai le confidenze per la classe target
    #     confidences = torch.softmax(logits, dim=1)[:, class_idx]  # (3 * L,)
        
    #     # Ridimensioniamo le confidenze e moltiplichiamo per la maschera finale
    #     # Aggiungiamo una dimensione extra per permettere la trasmissione del prodotto
    #     weighted_masks = final_mask * confidences.view(-1, 1)  # (3 * L, 196)
        
    #     # Estrai le confidenze per la classe target
    #     confidences = torch.softmax(logits, dim=1)[:, class_idx]  # (3 * L,)
        
    #     # Ridimensioniamo le confidenze e moltiplichiamo per la maschera finale
    #     # Aggiungiamo una dimensione extra per permettere la trasmissione del prodotto
    #     weighted_masks = final_mask * confidences.view(-1, 1)  # (3 * L, 196)
                
    #     # Somma i pesi per ogni token lungo la dimensione batch per ottenere una heatmap complessiva
    #     heatmap_ground_truth_class = torch.sum(weighted_masks, dim=0)  # Somma lungo la dimensione 0, ottieni (196,)
    #     # coverage_bias = torch.sum(final_mask, dim=0)  # Somma le maschere binarie lungo la dimensione 0
        
    #     # # Normalizza la heatmap rispetto alla copertura dei token
    #     # coverage_bias = torch.where(coverage_bias > 0, coverage_bias, 1)  # Evita la divisione per zero
    #     # heatmap_ground_truth_class = heatmap_ground_truth_class / coverage_bias  # Normalizzazione
    #     heatmap_ground_truth_class_reshape = heatmap_ground_truth_class.reshape((14, 14))  # Reshape in (14, 14)
            
    #     return heatmap_ground_truth_class_reshape
      

    def get_insertion_deletion(self, patch_perc, heatmap, image, baseline, label):
        """
        Generates confidence scores for insertion and deletion for the specif baseline and every patch_perc.

        Args:
            patch_perc (list): List of patch percentages to consider.
            heatmap (torch.Tensor): Original heatmap.
            image (torch.Tensor): Original image tensor.
            baseline (str): Baseline image type ('black', 'blur', 'random', or 'mean').
            label: True label of the image.

        Returns:
            dict: Dictionary containing confidence scores for 'insertion' and 'deletion' operations.
        """

        # Process the original image
        image = self.image_processor(images=image, return_tensors="pt")

        # Reshape and interpolate the heatmap to match the image size
        heatmap = heatmap.reshape((1, 1, 14, 14))
        gaussian_heatmap = interpolate(heatmap, size=(224, 224), mode='nearest')
        gaussian_heatmap = gaussian_heatmap[0, 0, :, :].to('cpu').detach()

        confidences = {}

        for operation in ['insertion', 'deletion']:
            batch_modified = []
            for percentage in patch_perc:
                modified_image = self.modify_image(operation=operation, heatmap=gaussian_heatmap, image=image,
                                                   percentage=percentage / 100, baseline=baseline, device=self.device)
                batch_modified.append(modified_image)

            batch_modified = torch.stack(batch_modified, dim=0).to(self.device)
            confidences[operation] = self.predict(batch_modified, label)

        return confidences

    def predict(self, obscured_inputs, true_class_index):
        """
        Predicts the class probabilities for the true class for a list of obscured inputs.

        Args:
            obscured_inputs (torch.Tensor): Batch of obscured images.
            true_class_index (int): True class index for the original image.

        Returns:
            list: List of predicted probabilities for the true class in each obscured input.
        """
        outputs = self.model(obscured_inputs)
        probabilities = F.softmax(outputs.logits, dim=1)

        true_class_probs = probabilities[:, true_class_index]

        return true_class_probs.tolist()


    def transpose_for_scores(self, x, num_masks):
        # Ridimensiona il tensore in (layers, seq_length, num_masks, attention_head_size)
        attention_head_size = int(x.size()[-1]/num_masks)
        new_x_shape = x.size()[:-1] + (num_masks, attention_head_size)
        x = x.view(new_x_shape)
        # Permuta per ottenere (layers, num_masks, seq_length, attention_head_size)
        return x.permute(0, 2, 1, 3)
