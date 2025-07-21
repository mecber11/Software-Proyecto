"""
Cargador de Datos Avanzado para Dataset SIPaKMeD
Sistema optimizado para PyTorch con CLAHE, augmentación y GPU

Características:
- Carga completa del dataset SIPaKMeD (5,015 imágenes .bmp)
- Normalización CLAHE para ajuste de contraste
- Augmentación avanzada: rotaciones (±15°), zoom, desplazamientos
- Segmentación para rotaciones
- Optimización GPU con DataLoader multiproceso
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from sklearn.model_selection import train_test_split
from collections import Counter

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLAHEProcessor:
    """Procesador CLAHE optimizado para imágenes cervicales"""
    
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def __call__(self, image):
        """Aplicar CLAHE a imagen"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Aplicar CLAHE
        enhanced = self.clahe.apply(gray)
        
        # Convertir de vuelta a RGB
        if len(image.shape) == 3:
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            enhanced_rgb = np.stack([enhanced] * 3, axis=-1)
        
        return enhanced_rgb

class SIPaKMeDDataset(Dataset):
    """
    Dataset personalizado para SIPaKMeD con preprocesamiento avanzado
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train',
        image_size: int = 224,
        apply_clahe: bool = True,
        augmentation: bool = True,
        validation_split: float = 0.2,
        random_state: int = 42
    ):
        """
        Args:
            root_dir: Directorio raíz del dataset SIPaKMeD
            split: 'train', 'val' o 'test'
            image_size: Tamaño de imagen de salida
            apply_clahe: Aplicar normalización CLAHE
            augmentation: Aplicar augmentación de datos
            validation_split: Porcentaje para validación
            random_state: Semilla para reproducibilidad
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.augmentation = augmentation and (split == 'train')
        
        # Mapeo de clases
        self.class_names = [
            'dyskeratotic', 
            'koilocytotic', 
            'metaplastic', 
            'parabasal', 
            'superficial_intermediate'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Cargar rutas de imágenes y etiquetas
        self.image_paths, self.labels = self._load_dataset()
        
        # Dividir en train/validation
        if split in ['train', 'val']:
            self.image_paths, self.labels = self._split_dataset(validation_split, random_state)
        
        # Configurar procesadores (no crear aquí para evitar problemas de pickle)
        self.apply_clahe_flag = apply_clahe
        self.transforms = self._get_transforms()
        
        logger.info(f"Dataset {split} cargado: {len(self.image_paths)} imágenes")
        logger.info(f"Distribución de clases: {Counter(self.labels)}")
    
    def _load_dataset(self) -> Tuple[List[str], List[int]]:
        """Cargar todas las imágenes .bmp y .dat del dataset SIPaKMeD"""
        image_paths = []
        labels = []
        
        total_images = 0
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directorio no encontrado: {class_dir}")
                continue
            
            # Buscar archivos .bmp (imágenes completas)
            bmp_files = list(class_dir.glob("*.bmp"))
            
            # Buscar archivos .dat (segmentaciones de células)
            dat_files = list(class_dir.glob("*.dat"))
            
            # Agregar archivos .bmp
            for img_path in bmp_files:
                image_paths.append(str(img_path))
                labels.append(self.class_to_idx[class_name])
            
            # Agregar archivos .dat (estos contienen coordenadas de células segmentadas)
            for dat_path in dat_files:
                # Para .dat, buscaremos la imagen .bmp correspondiente
                base_name = dat_path.stem.split('_')[0]  # Ej: 001_cyt01 -> 001
                corresponding_bmp = class_dir / f"{base_name}.bmp"
                
                if corresponding_bmp.exists():
                    # Agregar el .dat con referencia a su .bmp
                    image_paths.append(f"{dat_path}|{corresponding_bmp}")
                    labels.append(self.class_to_idx[class_name])
            
            class_count = len(bmp_files) + len(dat_files)
            total_images += class_count
            logger.info(f"Clase '{class_name}': {len(bmp_files)} .bmp + {len(dat_files)} .dat = {class_count} imágenes")
        
        logger.info(f"Total de imágenes cargadas: {total_images}")
        return image_paths, labels
    
    def _split_dataset(self, validation_split: float, random_state: int) -> Tuple[List[str], List[int]]:
        """Dividir dataset en train/validation manteniendo distribución de clases"""
        if len(self.image_paths) == 0:
            return [], []
        
        # Estratified split para mantener distribución de clases
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.image_paths, 
            self.labels,
            test_size=validation_split,
            random_state=random_state,
            stratify=self.labels
        )
        
        if self.split == 'train':
            return train_paths, train_labels
        else:  # validation
            return val_paths, val_labels
    
    def _get_transforms(self):
        """Configurar transformaciones según el split"""
        if self.split == 'train' and self.augmentation:
            # Augmentación agresiva para entrenamiento
            return A.Compose([
                # Transformaciones geométricas
                A.Rotate(limit=15, p=0.8, border_mode=cv2.BORDER_REFLECT),
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size), 
                    scale=(0.8, 1.0), 
                    ratio=(0.9, 1.1), 
                    p=0.8
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.2, 
                    rotate_limit=15, 
                    p=0.7,
                    border_mode=cv2.BORDER_REFLECT
                ),
                
                # Transformaciones de intensidad
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.6
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=20, 
                    p=0.3
                ),
                
                # Transformaciones de ruido y blur
                A.OneOf([
                    A.GaussNoise(p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.2),
                ], p=0.4),
                
                # Transformaciones espaciales avanzadas
                A.OneOf([
                    A.ElasticTransform(p=0.3),
                    A.GridDistortion(p=0.3),
                    A.OpticalDistortion(p=0.3),
                ], p=0.3),
                
                # Cutout y dropout  
                A.CoarseDropout(p=0.3),
                
                # Normalización final
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Solo transformaciones básicas para validación/test
            return A.Compose([
                A.Resize(height=self.image_size, width=self.image_size, p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Cargar y procesar una muestra (tanto .bmp como .dat)"""
        try:
            img_path = self.image_paths[idx]

            # Verificar si es un archivo .dat (formato: "path.dat|path.bmp")
            if '|' in img_path:
                dat_path, bmp_path = img_path.split('|')
                image = self._load_dat_segment(dat_path, bmp_path)
            else:
                # Es un archivo .bmp normal
                image = cv2.imread(img_path)
                if image is None:
                    logger.error(f"No se pudo cargar la imagen: {img_path}")
                    # Retornar imagen en negro como fallback
                    image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                else:
                    # Convertir de BGR a RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Redimensionar a tamaño estándar para garantizar consistencia
            if image.shape[:2] != (self.image_size, self.image_size):
                image = cv2.resize(image, (self.image_size, self.image_size))

            # Aplicar CLAHE si está habilitado (crear cada vez para evitar problemas de pickle)
            if self.apply_clahe_flag:
                clahe_processor = CLAHEProcessor()
                image = clahe_processor(image)

            # Aplicar transformaciones
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented['image']
            else:
                # Fallback si no hay transforms
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            label = self.labels[idx]

            return image, label

        except Exception as e:
            logger.error(f"Error procesando imagen {idx}: {e}")
            # Retornar tensor vacío y label 0 como fallback
            return torch.zeros(3, self.image_size, self.image_size), 0
    
    def _load_dat_segment(self, dat_path: str, bmp_path: str) -> np.ndarray:
        """Cargar segmento de célula desde archivo .dat"""
        try:
            # Cargar imagen completa
            full_image = cv2.imread(bmp_path)
            if full_image is None:
                return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            
            # Leer coordenadas del archivo .dat
            with open(dat_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                # Si no hay suficientes coordenadas, usar imagen completa redimensionada
                return cv2.resize(full_image, (self.image_size, self.image_size))
            
            # Extraer coordenadas (formato típico de SIPaKMeD)
            coords = []
            for line in lines:
                if line.strip():
                    try:
                        x, y = map(float, line.strip().split())
                        coords.append((int(x), int(y)))
                    except:
                        continue
            
            if len(coords) < 3:
                # Si no hay suficientes coordenadas válidas, usar imagen completa
                return cv2.resize(full_image, (self.image_size, self.image_size))
            
            # Calcular bounding box de la célula
            coords = np.array(coords)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            
            # Agregar padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(full_image.shape[1], x_max + padding)
            y_max = min(full_image.shape[0], y_max + padding)
            
            # Extraer segmento
            segment = full_image[y_min:y_max, x_min:x_max]
            
            # Redimensionar a tamaño estándar SIEMPRE
            if segment.size > 0:
                segment = cv2.resize(segment, (self.image_size, self.image_size))
                return segment
            else:
                return cv2.resize(full_image, (self.image_size, self.image_size))
                
        except Exception as e:
            logger.warning(f"Error cargando segmento .dat {dat_path}: {e}")
            # Fallback: usar imagen completa siempre redimensionada
            try:
                full_image = cv2.imread(bmp_path)
                if full_image is not None:
                    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
                    return cv2.resize(full_image, (self.image_size, self.image_size))
            except:
                pass
            
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

def get_data_loaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    validation_split: float = 0.2,
    apply_clahe: bool = True,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Crear DataLoaders para entrenamiento y validación
    
    Args:
        root_dir: Directorio del dataset SIPaKMeD
        batch_size: Tamaño de batch
        image_size: Tamaño de imagen
        num_workers: Número de workers para carga de datos
        pin_memory: Usar pin memory para GPU
        validation_split: Porcentaje para validación
        apply_clahe: Aplicar normalización CLAHE
        random_state: Semilla para reproducibilidad
    
    Returns:
        Tuple de (train_loader, val_loader)
    """
    
    # Crear datasets
    train_dataset = SIPaKMeDDataset(
        root_dir=root_dir,
        split='train',
        image_size=image_size,
        apply_clahe=apply_clahe,
        augmentation=True,
        validation_split=validation_split,
        random_state=random_state
    )
    
    val_dataset = SIPaKMeDDataset(
        root_dir=root_dir,
        split='val',
        image_size=image_size,
        apply_clahe=apply_clahe,
        augmentation=False,
        validation_split=validation_split,
        random_state=random_state
    )
    
    # Crear DataLoaders (sin multiprocessing para evitar problemas de pickle)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Forzar a 0 para evitar problemas de pickle
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Forzar a 0 para evitar problemas de pickle
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=False
    )
    
    return train_loader, val_loader

def calculate_dataset_stats(root_dir: str, sample_size: int = 1000):
    """Calcular estadísticas del dataset para normalización"""
    logger.info(f"Calculando estadísticas del dataset (muestra de {sample_size} imágenes)...")
    
    # Cargar muestra de imágenes
    all_images = []
    for class_name in ['dyskeratotic', 'koilocytotic', 'metaplastic', 'parabasal', 'superficial_intermediate']:
        class_dir = Path(root_dir) / class_name
        if class_dir.exists():
            bmp_files = list(class_dir.glob("*.bmp"))
            all_images.extend(bmp_files)
    
    # Seleccionar muestra aleatoria
    if len(all_images) > sample_size:
        all_images = random.sample(all_images, sample_size)
    
    # Calcular estadísticas
    pixel_values = []
    clahe_processor = CLAHEProcessor()
    
    for img_path in all_images:
        try:
            image = cv2.imread(str(img_path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = clahe_processor(image)
                pixel_values.append(image.flatten())
        except Exception as e:
            logger.warning(f"Error procesando {img_path}: {e}")
    
    if pixel_values:
        all_pixels = np.concatenate(pixel_values)
        mean = np.mean(all_pixels) / 255.0
        std = np.std(all_pixels) / 255.0
        
        logger.info(f"Media: {mean:.4f}, Desviación estándar: {std:.4f}")
        return mean, std
    else:
        logger.warning("No se pudieron calcular estadísticas, usando valores por defecto")
        return 0.5, 0.5

def get_class_weights(root_dir: str) -> torch.Tensor:
    """Calcular pesos de clase para manejar desbalance"""
    class_counts = {}
    class_names = ['dyskeratotic', 'koilocytotic', 'metaplastic', 'parabasal', 'superficial_intermediate']
    
    for class_name in class_names:
        class_dir = Path(root_dir) / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.bmp")))
            class_counts[class_name] = count
        else:
            class_counts[class_name] = 0
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_names)
    
    # Calcular pesos inversamente proporcionales a la frecuencia
    weights = []
    for class_name in class_names:
        weight = total_samples / (num_classes * class_counts[class_name]) if class_counts[class_name] > 0 else 0
        weights.append(weight)
    
    weights_tensor = torch.FloatTensor(weights)
    
    logger.info("Distribución de clases:")
    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name}: {class_counts[class_name]} muestras (peso: {weights[i]:.3f})")
    
    return weights_tensor

if __name__ == "__main__":
    # Prueba del cargador de datos
    root_dir = "C:/Users/David/Downloads/SIPaKMeD"
    
    print("🔍 Probando cargador de datos SIPaKMeD...")
    
    # Calcular estadísticas
    mean, std = calculate_dataset_stats(root_dir)
    
    # Calcular pesos de clase
    class_weights = get_class_weights(root_dir)
    
    # Crear DataLoaders
    train_loader, val_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=8,  # Pequeño para prueba
        image_size=224,
        num_workers=2,
        validation_split=0.2
    )
    
    print(f"✅ DataLoaders creados:")
    print(f"   Training: {len(train_loader)} batches, {len(train_loader.dataset)} muestras")
    print(f"   Validation: {len(val_loader)} batches, {len(val_loader.dataset)} muestras")
    
    # Probar carga de un batch
    print("\n🔄 Probando carga de batch...")
    try:
        images, labels = next(iter(train_loader))
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels únicos: {torch.unique(labels)}")
        print(f"   Rango de píxeles: [{images.min():.3f}, {images.max():.3f}]")
        print("✅ Carga de datos exitosa!")
    except Exception as e:
        print(f"❌ Error en carga de datos: {e}")