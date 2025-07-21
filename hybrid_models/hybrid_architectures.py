"""
Arquitecturas de Modelos Híbridos para Clasificación de Células Cervicales
Sistema SIPaKMeD con PyTorch y GPU

Implementa dos metodologías híbridas avanzadas:
1. HybridEnsembleCNN - Fusión de múltiples CNNs pre-entrenadas con atención
2. HybridMultiScaleCNN - CNN personalizada con bloques multi-escala
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights
import math

class ChannelAttention(nn.Module):
    """Módulo de atención por canales para mejorar características relevantes"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Módulo de atención espacial para enfocarse en regiones importantes"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class MultiScaleBlock(nn.Module):
    """Bloque multi-escala para capturar características a diferentes resoluciones"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        
        # Diferentes escalas de convolución
        self.branch1x1 = nn.Conv2d(in_channels, out_channels//4, 1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.Conv2d(out_channels//4, out_channels//4, 3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.Conv2d(out_channels//4, out_channels//4, 3, padding=1),
            nn.Conv2d(out_channels//4, out_channels//4, 3, padding=1)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, 1)
        )
        
        self.conv_reduce = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)
        outputs = self.conv_reduce(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        outputs = self.cbam(outputs)
        
        return outputs

class HybridEnsembleCNN(nn.Module):
    """
    Modelo Híbrido 1: Ensemble de CNNs Pre-entrenadas con Atención
    
    Combina ResNet50, MobileNetV2 y EfficientNet-B0 usando mecanismos de atención
    para obtener una representación enriquecida de las características celulares.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(HybridEnsembleCNN, self).__init__()
        
        # Cargar modelos pre-entrenados
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Extraer características sin clasificadores finales
        self.resnet_features = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.mobilenet_features = self.mobilenet.features
        self.efficientnet_features = self.efficientnet.features
        
        # Dimensiones de características de cada modelo
        self.resnet_feat_dim = 2048
        self.mobilenet_feat_dim = 1280
        self.efficientnet_feat_dim = 1280
        
        # Adaptadores para uniformizar dimensiones
        self.resnet_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(self.resnet_feat_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.mobilenet_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(self.mobilenet_feat_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.efficientnet_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(self.efficientnet_feat_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Módulo de atención para fusionar características
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(1536, 512, 3, padding=1),  # 512 * 3 = 1536
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Clasificador final con dropout múltiple
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Inicialización de pesos para capas nuevas
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos de las capas nuevas"""
        for m in [self.resnet_adapter, self.mobilenet_adapter, self.efficientnet_adapter, 
                  self.attention_fusion, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Extraer características de cada modelo
        resnet_feat = self.resnet_features(x)
        mobilenet_feat = self.mobilenet_features(x)
        efficientnet_feat = self.efficientnet_features(x)
        
        # Adaptar características a dimensiones uniformes
        resnet_adapted = self.resnet_adapter(resnet_feat)
        mobilenet_adapted = self.mobilenet_adapter(mobilenet_feat)
        efficientnet_adapted = self.efficientnet_adapter(efficientnet_feat)
        
        # Fusionar características usando atención
        fused_features = torch.cat([resnet_adapted, mobilenet_adapted, efficientnet_adapted], dim=1)
        attended_features = self.attention_fusion(fused_features)
        
        # Clasificación final
        output = self.classifier(attended_features)
        
        return output

class HybridMultiScaleCNN(nn.Module):
    """
    Modelo Híbrido 2: CNN Multi-Escala Personalizada
    
    Arquitectura personalizada que procesa características a múltiples escalas
    usando bloques multi-escala inspirados en Inception con módulos CBAM.
    """
    def __init__(self, num_classes=5):
        super(HybridMultiScaleCNN, self).__init__()
        
        # Stem: Entrada y características iniciales
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Bloques multi-escala progresivos
        self.stage1 = nn.Sequential(
            MultiScaleBlock(64, 128),
            MultiScaleBlock(128, 128),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.stage2 = nn.Sequential(
            MultiScaleBlock(128, 256),
            MultiScaleBlock(256, 256),
            MultiScaleBlock(256, 256),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.stage3 = nn.Sequential(
            MultiScaleBlock(256, 512),
            MultiScaleBlock(512, 512),
            MultiScaleBlock(512, 512),
            MultiScaleBlock(512, 512),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.stage4 = nn.Sequential(
            MultiScaleBlock(512, 1024),
            MultiScaleBlock(1024, 1024),
            MultiScaleBlock(1024, 1024),
        )
        
        # Atención global
        self.global_attention = CBAM(1024)
        
        # Clasificador con atención espacial adaptativa
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos usando He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Procesamiento progresivo con bloques multi-escala
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Atención global
        x = self.global_attention(x)
        
        # Pooling adaptativo y clasificación
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        
        return x

def get_hybrid_model(model_type="ensemble", num_classes=5, pretrained=True):
    """
    Factory function para crear modelos híbridos
    
    Args:
        model_type (str): "ensemble" o "multiscale"
        num_classes (int): Número de clases (default: 5 para SIPaKMeD)
        pretrained (bool): Usar pesos pre-entrenados para ensemble
    
    Returns:
        torch.nn.Module: Modelo híbrido inicializado
    """
    if model_type.lower() == "ensemble":
        return HybridEnsembleCNN(num_classes=num_classes, pretrained=pretrained)
    elif model_type.lower() == "multiscale":
        return HybridMultiScaleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}. Use 'ensemble' o 'multiscale'")

def count_parameters(model):
    """Contar parámetros del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Pruebas rápidas de los modelos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # Probar Hybrid Ensemble CNN
    print("\n=== HYBRID ENSEMBLE CNN ===")
    model_ensemble = get_hybrid_model("ensemble", num_classes=5)
    model_ensemble = model_ensemble.to(device)
    
    x = torch.randn(2, 3, 224, 224).to(device)
    output = model_ensemble(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parámetros entrenables: {count_parameters(model_ensemble):,}")
    
    # Probar Hybrid Multi-Scale CNN
    print("\n=== HYBRID MULTI-SCALE CNN ===")
    model_multiscale = get_hybrid_model("multiscale", num_classes=5)
    model_multiscale = model_multiscale.to(device)
    
    output = model_multiscale(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parámetros entrenables: {count_parameters(model_multiscale):,}")
    
    print(f"\n✅ Ambos modelos híbridos funcionan correctamente en {device}")