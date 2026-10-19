import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
import time
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator

# Set global font to Times New Roman (English only)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==================== 1. ENHANCED DATA PREPROCESSING ====================
class EnhancedTabularDataset(Dataset):
    """Enhanced dataset with multiple augmentation techniques"""
    
    def __init__(self, features, targets, augment=False, augment_prob=0.3):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.augment = augment
        self.augment_prob = augment_prob
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx].clone()
        targets = self.targets[idx].clone()
        
        if self.augment and torch.rand(1).item() < self.augment_prob:
            # Multiple augmentation strategies
            aug_type = torch.randint(0, 4, (1,)).item()
            
            if aug_type == 0:
                # Gaussian noise
                noise = torch.randn_like(features) * 0.02
                features = features + noise
            elif aug_type == 1:
                # Feature masking (random dropout of features)
                mask = torch.rand_like(features) > 0.1
                features = features * mask
            elif aug_type == 2:
                # Feature scaling
                scale = torch.randn(1).item() * 0.05 + 1.0
                features = features * scale
            else:
                # Mixup with random sample
                idx2 = torch.randint(0, len(self), (1,)).item()
                alpha = torch.rand(1).item() * 0.2
                features = (1 - alpha) * features + alpha * self.features[idx2]
                targets = (1 - alpha) * targets + alpha * self.targets[idx2]
        
        return features, targets


def enhanced_preprocessing(filepath='data.csv', test_size=0.1, val_size=0.15, random_state=42):
    """Enhanced preprocessing with feature engineering"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Data shape: {df.shape}")
    
    # Assume 8 input features and 4 outputs
    input_cols = df.columns[:8].tolist()
    # Last 4 columns as outputs
    output_cols = df.columns[-4:].tolist()
    
    # Convert to string
    input_cols = [str(col) for col in input_cols]
    output_cols = [str(col) for col in output_cols]
    
    print(f"Input features: {input_cols}")
    print(f"Output targets: {output_cols}")
    
    X = df[input_cols].values
    y = df[output_cols].values.reshape(-1, 4)  # Ensure y is 2D array with 4 outputs
    
    # Simplified feature engineering to reduce memory usage
    print("\nPerforming simplified feature engineering...")
    X_engineered = X.copy()
    
    # 1. Add only square terms to reduce feature count
    n_original = X.shape[1]
    for i in range(n_original):
        # Add only square terms, no interaction terms
        X_engineered = np.column_stack([X_engineered, X[:, i]**2])
    
    # 2. Add only mean and standard deviation statistical features
    X_engineered = np.column_stack([X_engineered, np.mean(X, axis=1, keepdims=True)])
    X_engineered = np.column_stack([X_engineered, np.std(X, axis=1, keepdims=True)])
    
    print(f"Original features: {X.shape[1]}")
    print(f"Engineered features: {X_engineered.shape[1]}")
    
    # Use simpler scaler
    X_scaler = StandardScaler()  # Use StandardScaler
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X_engineered)
    y_scaled = y_scaler.fit_transform(y)
    
    # Create dataset with augmentation
    full_dataset = EnhancedTabularDataset(X_scaled, y_scaled, augment=False)
    
    # Split dataset
    total_size = len(full_dataset)
    test_size_int = int(total_size * test_size)
    val_size_int = int(total_size * val_size)
    train_size_int = total_size - test_size_int - val_size_int
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size_int, val_size_int, test_size_int],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    # Enable augmentation for training only
    train_dataset.dataset.augment = True
    
    # Data statistics
    data_stats = {
        'input_cols': input_cols,
        'output_cols': output_cols,
        'n_original_features': X.shape[1],
        'n_engineered_features': X_engineered.shape[1],
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'X_original': X,  # Save original data for later use
        'y_original': y   # Save original data for later use
    }
    
    return (train_dataset, val_dataset, test_dataset,
            X_scaler, y_scaler, input_cols, output_cols, data_stats, X_engineered.shape[1])


# ==================== 2. SIMPLIFIED BUT POWERFUL MODEL ====================
class ResidualBlock(nn.Module):
    """Simplified but effective residual block"""
    
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features * 2),  # Reduce parameters
            nn.BatchNorm1d(out_features * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features * 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7)
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        # Final activation
        self.final_activation = nn.GELU()
    
    def forward(self, x):
        residual = self.skip_connection(x)
        main = self.main_path(x)
        out = main + residual
        out = self.final_activation(out)
        return out


class HighAccuracyRegressor(nn.Module):
    """Simplified but powerful regressor for high accuracy"""
    
    def __init__(self, input_dim, output_dim=4,  # 4 outputs
                 base_dim=256, num_blocks=8, dropout_rate=0.3):  # Reduce model size
        super(HighAccuracyRegressor, self).__init__()
        
        self.input_dim = input_dim
        self.base_dim = base_dim
        
        # Initial projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.BatchNorm1d(base_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Residual blocks with progressive dimension change
        self.blocks = nn.ModuleList()
        current_dim = base_dim
        
        # Define dimension progression
        dim_progression = []
        for i in range(num_blocks):
            if i < num_blocks // 3:
                # Increase dimension
                next_dim = int(current_dim * 1.2)  # Reduce growth rate
            elif i < 2 * num_blocks // 3:
                # Maintain dimension
                next_dim = current_dim
            else:
                # Decrease dimension towards output
                next_dim = max(int(current_dim * 0.8), output_dim * 4)
            
            dim_progression.append(next_dim)
            current_dim = next_dim
        
        # Create blocks
        prev_dim = base_dim
        for next_dim in dim_progression:
            block = ResidualBlock(prev_dim, next_dim, dropout_rate)
            self.blocks.append(block)
            prev_dim = next_dim
        
        self.final_dim = prev_dim
        
        # Simplified output layer
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(prev_dim // 2, output_dim)
        )
        
        # Weight initialization
        self._initialize_weights()
        
        # Print info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"HighAccuracyRegressor created with {total_params:,} parameters")
        print(f"Architecture: Input({input_dim}) -> {base_dim} -> {num_blocks} blocks -> Output({output_dim})")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial projection
        x = self.input_layer(x)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x)
        
        # Final output
        output = self.output_layer(x)
        
        return output
    
    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'base_dim': self.base_dim,
            'num_blocks': len(self.blocks),
            'dropout_rate': 0.3
        }


# ==================== 3. IMPROVED TRAINING WITH ENSEMBLE ====================
class EnsembleModel(nn.Module):
    """Ensemble of models for better accuracy"""
    
    def __init__(self, input_dim, output_dim=4, n_models=2):  # 4 outputs
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList([
            HighAccuracyRegressor(input_dim, output_dim, 
                                 base_dim=256 + i*32,  # Reduce base dimension
                                 num_blocks=8 + i)  # Reduce number of blocks
            for i in range(n_models)
        ])
        
        # Weighted combination
        self.weights = nn.Parameter(torch.ones(n_models) / n_models)
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        outputs = torch.stack(outputs, dim=0)
        weights = F.softmax(self.weights, dim=0)
        
        # Weighted average
        weighted_output = torch.sum(outputs * weights.view(-1, 1, 1), dim=0)
        
        return weighted_output


class ImprovedTrainer:
    """Improved trainer with better strategies"""
    
    def __init__(self, model, output_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.output_names = [str(name) for name in output_names]
        self.best_val_loss = float('inf')
        self.best_val_r2 = -float('inf')
        self.best_model_state = None
        self.start_time = time.time()
        
        # Record training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rate': [],
            'grad_norm': [],
            'time_elapsed': []
        }
        
        print(f"\n{'='*80}")
        print(f"IMPROVED TRAINER INITIALIZED")
        print(f"{'='*80}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Device: {device}")
    
    def calculate_r2_scores(self, outputs, targets):
        """Calculate R² scores for each output"""
        outputs_np = outputs.detach().cpu().numpy() if torch.is_tensor(outputs) else outputs
        targets_np = targets.detach().cpu().numpy() if torch.is_tensor(targets) else targets
        
        r2_scores = {}
        for i, name in enumerate(self.output_names):
            try:
                r2 = r2_score(targets_np[:, i], outputs_np[:, i])
                r2_scores[name] = max(r2, -1.0)
            except:
                r2_scores[name] = -1.0
        
        if r2_scores:
            r2_scores['overall'] = np.mean(list(r2_scores.values()))
        
        return r2_scores
    
    def train_epoch(self, dataloader, criterion, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []
        grad_norm_total = 0
        grad_norm_count = 0
        
        accumulation_steps = 2  # Reduce gradient accumulation steps
        optimizer.zero_grad()
        
        for batch_idx, (features, targets) in enumerate(dataloader):
            features, targets = features.to(self.device), targets.to(self.device)
            
            outputs = self.model(features)
            
            # Combined loss for better stability
            mse_loss = criterion(outputs, targets)
            l1_loss = F.l1_loss(outputs, targets)
            loss = 0.8 * mse_loss + 0.2 * l1_loss
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Calculate gradient norm
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norm_total += total_norm
                grad_norm_count += 1
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
            
            total_loss += loss.item() * accumulation_steps
            all_outputs.append(outputs.detach().cpu())  # Move to CPU
            all_targets.append(targets.detach().cpu())  # Move to CPU
        
        if len(dataloader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        r2_scores = self.calculate_r2_scores(all_outputs, all_targets)
        
        # Calculate average gradient norm
        avg_grad_norm = grad_norm_total / grad_norm_count if grad_norm_count > 0 else 0
        
        return total_loss / len(dataloader), r2_scores, avg_grad_norm
    
    def evaluate(self, dataloader, criterion):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in dataloader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                
                mse_loss = criterion(outputs, targets)
                l1_loss = F.l1_loss(outputs, targets)
                loss = 0.8 * mse_loss + 0.2 * l1_loss
                
                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        r2_scores = self.calculate_r2_scores(all_outputs, all_targets)
        
        return total_loss / len(dataloader), r2_scores, all_outputs.numpy(), all_targets.numpy()
    
    def train(self, train_loader, val_loader, criterion, optimizer, scheduler,
              num_epochs=1000, patience=100, model_save_path='best_model.pth'):
        
        print(f"\n{'='*80}")
        print(f"STARTING TRAINING")
        print(f"{'='*80}")
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_r2, grad_norm = self.train_epoch(train_loader, criterion, optimizer, scheduler)
            
            # Validation
            val_loss, val_r2, _, _ = self.evaluate(val_loader, criterion)
            
            current_lr = optimizer.param_groups[0]['lr']
            current_r2 = val_r2.get('overall', -1)
            
            # Record training history
            epoch_time = time.time() - epoch_start_time
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_r2'].append(train_r2.get('overall', 0))
            self.history['val_r2'].append(current_r2)
            self.history['learning_rate'].append(current_lr)
            self.history['grad_norm'].append(grad_norm)
            self.history['time_elapsed'].append(epoch_time)
            
            # Save best model
            if current_r2 > self.best_val_r2:
                self.best_val_r2 = current_r2
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_r2': val_r2,
                    'best_r2': self.best_val_r2,
                    'history': self.history
                }, model_save_path)
                
                patience_counter = 0
                print(f"\n✓ Epoch {epoch+1:04d}: New best model (R²: {current_r2:.6f})")
            else:
                patience_counter += 1
            
            # Progress report
            if (epoch + 1) % 25 == 0 or epoch == 0:
                elapsed = time.time() - self.start_time
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                
                print(f"\nEpoch {epoch+1:04d}/{num_epochs} | Time: {int(hours)}h {int(minutes)}m | Epoch time: {epoch_time:.2f}s")
                print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                print(f"  Train R²: {train_r2.get('overall', 0):.6f} | Val R²: {current_r2:.6f}")
                print(f"  Best R²: {self.best_val_r2:.6f} (epoch {best_epoch+1})")
                print(f"  LR: {current_lr:.2e} | Grad Norm: {grad_norm:.4f} | Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.best_val_r2


# ==================== 4. OPTIMIZED MAIN PIPELINE ====================
def run_optimized_pipeline():
    """Run optimized pipeline for high R²"""
    
    # Configuration - reduce memory usage
    config = {
        'data_path': 'data.csv',
        'use_ensemble': False,  # Disable ensemble to save memory
        'batch_size': 64,  # Reduce batch size
        'learning_rate': 0.001,
        'num_epochs': 800,  # Reduce training epochs
        'patience': 100,
        'weight_decay': 1e-6,
        'results_dir': 'high_accuracy_results',
        'target_r2': 0.95
    }
    
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Check GPU memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"\n{'='*80}")
    print(f"HIGH ACCURACY TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 1. Data preprocessing
    print(f"\n[1/4] LOADING AND PREPROCESSING DATA")
    train_dataset, val_dataset, test_dataset, X_scaler, y_scaler, \
    input_cols, output_cols, data_stats, input_dim = enhanced_preprocessing(config['data_path'])
    
    print(f"\nDataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Input dimension: {input_dim}")
    
    # 2. Create model
    print(f"\n[2/4] CREATING MODEL")
    if config['use_ensemble']:
        model = EnsembleModel(input_dim, output_dim=4, n_models=2)
        print("Using ensemble of 2 models")
    else:
        model = HighAccuracyRegressor(input_dim, output_dim=4, base_dim=256, num_blocks=8)
        print("Using single high-accuracy model")
    
    # 3. Setup data loaders and training
    print(f"\n[3/4] SETTING UP TRAINING")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    trainer = ImprovedTrainer(model, output_cols, device)
    
    # 4. Train
    print(f"\n[4/4] TRAINING MODEL")
    best_val_r2 = trainer.train(
        train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=config['num_epochs'],
        patience=config['patience'],
        model_save_path=os.path.join(config['results_dir'], 'best_model.pth')
    )
    
    # 5. Generate training visualizations
    print(f"\n[5/4] GENERATING TRAINING VISUALIZATIONS")
    generate_training_visualizations(trainer.history, config['results_dir'])
    
    # 6. Final evaluation on all datasets
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION ON ALL DATASETS")
    print(f"{'='*80}")
    
    # Create separate loaders without augmentation for evaluation
    train_eval_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_eval_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Evaluate on all datasets
    train_loss, train_r2, train_outputs_scaled, train_targets_scaled = trainer.evaluate(train_eval_loader, criterion)
    val_loss, val_r2, val_outputs_scaled, val_targets_scaled = trainer.evaluate(val_eval_loader, criterion)
    test_loss, test_r2, test_outputs_scaled, test_targets_scaled = trainer.evaluate(test_loader, criterion)
    
    # Inverse transform to get original scale values
    train_outputs = y_scaler.inverse_transform(train_outputs_scaled)
    train_targets = y_scaler.inverse_transform(train_targets_scaled)
    val_outputs = y_scaler.inverse_transform(val_outputs_scaled)
    val_targets = y_scaler.inverse_transform(val_targets_scaled)
    test_outputs = y_scaler.inverse_transform(test_outputs_scaled)
    test_targets = y_scaler.inverse_transform(test_targets_scaled)
    
    # Print results
    print(f"\nDataset Results:")
    print(f"{'Dataset':<15} {'Samples':<10} {'Loss':<12} {'R²':<10}")
    print(f"{'-'*50}")
    print(f"{'Training':<15} {len(train_outputs):<10} {train_loss:<12.8f} {train_r2.get('overall', 0):<10.6f}")
    print(f"{'Validation':<15} {len(val_outputs):<10} {val_loss:<12.8f} {val_r2.get('overall', 0):<10.6f}")
    print(f"{'Test':<15} {len(test_outputs):<10} {test_loss:<12.8f} {test_r2.get('overall', 0):<10.6f}")
    
    # Print per-output R²
    print(f"\nPer-Output R² Scores:")
    print(f"{'Output':<20} {'Train':<12} {'Val':<12} {'Test':<12}")
    print(f"{'-'*56}")
    for name in output_cols:
        print(f"{name:<20} {train_r2.get(name, 0):<12.6f} {val_r2.get(name, 0):<12.6f} {test_r2.get(name, 0):<12.6f}")
    
    # Check target achievement
    overall_test_r2 = test_r2.get('overall', 0)
    if overall_test_r2 >= config['target_r2']:
        print(f"\n🎉🎉🎉 TARGET ACHIEVED! Test R² = {overall_test_r2:.6f} >= {config['target_r2']} 🎉🎉🎉")
    else:
        print(f"\n⚠️  Target not reached. Current Test R²: {overall_test_r2:.6f}, Target: {config['target_r2']}")
    
    # Save results
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}")
    
    # Save model package with all datasets predictions
    model_package = {
        'model_state_dict': model.state_dict(),
        'feature_scaler': X_scaler,
        'target_scaler': y_scaler,
        'input_cols': input_cols,
        'output_cols': output_cols,
        'data_stats': data_stats,
        'train_metrics': train_r2,
        'val_metrics': val_r2,
        'test_metrics': test_r2,
        'training_history': trainer.history
    }
    
    torch.save(model_package, os.path.join(config['results_dir'], 'model_package.pth'))
    
    # Save predictions for all datasets
    save_all_predictions(
        train_outputs, train_targets, val_outputs, val_targets, 
        test_outputs, test_targets, output_cols, config['results_dir']
    )
    
    # Save training history
    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(os.path.join(config['results_dir'], 'training_history.csv'), index=False)
    
    # Generate comparison visualizations
    generate_train_test_comparison(
        train_outputs, train_targets, val_outputs, val_targets,
        test_outputs, test_targets, output_cols, config['results_dir']
    )
    
    print(f"\n✅ Results saved to '{config['results_dir']}'")
    
    return model, trainer, test_r2


def save_all_predictions(train_outputs, train_targets, val_outputs, val_targets, 
                        test_outputs, test_targets, output_cols, save_dir):
    """Save predictions for all datasets (training, validation, test)"""
    
    # Create DataFrames for each dataset
    train_results = pd.DataFrame(train_outputs, columns=[f'pred_{col}' for col in output_cols])
    val_results = pd.DataFrame(val_outputs, columns=[f'pred_{col}' for col in output_cols])
    test_results = pd.DataFrame(test_outputs, columns=[f'pred_{col}' for col in output_cols])
    
    for i, col in enumerate(output_cols):
        col_str = str(col)
        
        # Training set
        train_results[f'true_{col_str}'] = train_targets[:, i]
        train_results[f'error_{col_str}'] = train_outputs[:, i] - train_targets[:, i]
        train_results['dataset'] = 'training'
        
        # Validation set
        val_results[f'true_{col_str}'] = val_targets[:, i]
        val_results[f'error_{col_str}'] = val_outputs[:, i] - val_targets[:, i]
        val_results['dataset'] = 'validation'
        
        # Test set
        test_results[f'true_{col_str}'] = test_targets[:, i]
        test_results[f'error_{col_str}'] = test_outputs[:, i] - test_targets[:, i]
        test_results['dataset'] = 'test'
    
    # Save individual datasets
    train_results.to_csv(os.path.join(save_dir, 'train_predictions.csv'), index=False)
    val_results.to_csv(os.path.join(save_dir, 'val_predictions.csv'), index=False)
    test_results.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)
    
    # Combine all datasets
    all_results = pd.concat([train_results, val_results, test_results], ignore_index=True)
    all_results.to_csv(os.path.join(save_dir, 'all_predictions.csv'), index=False)
    
    print(f"  ✓ Predictions saved for all datasets:")
    print(f"    - Training: {len(train_results)} samples")
    print(f"    - Validation: {len(val_results)} samples")
    print(f"    - Test: {len(test_results)} samples")
    print(f"    - Combined: {len(all_results)} samples")


def generate_train_test_comparison(train_outputs, train_targets, val_outputs, val_targets,
                                  test_outputs, test_targets, output_cols, save_dir):
    """Generate visualizations comparing training and test sets"""
    
    plot_dir = os.path.join(save_dir, 'comparison_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for i, name in enumerate(output_cols):
        name_str = str(name)
        
        # Prepare data
        train_pred = train_outputs[:, i]
        train_true = train_targets[:, i]
        val_pred = val_outputs[:, i]
        val_true = val_targets[:, i]
        test_pred = test_outputs[:, i]
        test_true = test_targets[:, i]
        
        # 1. Combined scatter plot (Train, Val, Test in one plot)
        plt.figure(figsize=(14, 12))
        
        # Plot each dataset with different colors
        plt.scatter(train_true, train_pred, alpha=0.6, s=40, c='blue', label='Training Set', edgecolors='black', linewidth=0.5)
        plt.scatter(val_true, val_pred, alpha=0.6, s=40, c='orange', label='Validation Set', edgecolors='black', linewidth=0.5)
        plt.scatter(test_true, test_pred, alpha=0.6, s=40, c='green', label='Test Set', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        all_true = np.concatenate([train_true, val_true, test_true])
        all_pred = np.concatenate([train_pred, val_pred, test_pred])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=3)
        
        # Calculate R² for each dataset
        train_r2 = r2_score(train_true, train_pred)
        val_r2 = r2_score(val_true, val_pred)
        test_r2 = r2_score(test_true, test_pred)
        
        plt.title(f'Predicted vs True Values: {name_str}\nTraining vs Validation vs Test Sets', 
                  fontsize=24, fontweight='bold')
        plt.xlabel('True Values', fontsize=20)
        plt.ylabel('Predicted Values', fontsize=20)
        plt.legend(fontsize=16, loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add R² information
        stats_text = f'Training R² = {train_r2:.4f}\nValidation R² = {val_r2:.4f}\nTest R² = {test_r2:.4f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=16,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'combined_scatter_{name_str}.png'), dpi=400, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Combined scatter plot saved: combined_scatter_{name_str}.png")
        
        # 2. Error distribution comparison
        plt.figure(figsize=(14, 10))
        
        train_errors = train_pred - train_true
        val_errors = val_pred - val_true
        test_errors = test_pred - test_true
        
        # Create box plot for error distribution
        data_to_plot = [train_errors, val_errors, test_errors]
        labels = ['Training', 'Validation', 'Test']
        
        bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
        
        # Customize colors
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        plt.title(f'Error Distribution Comparison: {name_str}', fontsize=24, fontweight='bold')
        plt.xlabel('Dataset', fontsize=20)
        plt.ylabel('Prediction Error', fontsize=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add error statistics
        stats_text = f'Training Error: Mean={train_errors.mean():.4f}, Std={train_errors.std():.4f}\n'
        stats_text += f'Validation Error: Mean={val_errors.mean():.4f}, Std={val_errors.std():.4f}\n'
        stats_text += f'Test Error: Mean={test_errors.mean():.4f}, Std={test_errors.std():.4f}'
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'error_comparison_{name_str}.png'), dpi=400, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Error comparison plot saved: error_comparison_{name_str}.png")
        
        # 3. Performance metrics comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        # Subplot 1: R² comparison
        datasets = ['Training', 'Validation', 'Test']
        r2_values = [train_r2, val_r2, test_r2]
        colors = ['blue', 'orange', 'green']
        
        bars = ax1.bar(datasets, r2_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Dataset', fontsize=16)
        ax1.set_ylabel('R² Score', fontsize=16)
        ax1.set_title('R² Score Comparison', fontsize=18, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, r2_val in zip(bars, r2_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2_val:.4f}', ha='center', va='bottom', fontsize=14)
        
        # Subplot 2: RMSE comparison
        train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
        rmse_values = [train_rmse, val_rmse, test_rmse]
        
        bars = ax2.bar(datasets, rmse_values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Dataset', fontsize=16)
        ax2.set_ylabel('RMSE', fontsize=16)
        ax2.set_title('RMSE Comparison', fontsize=18, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rmse_val in zip(bars, rmse_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rmse_val:.4f}', ha='center', va='bottom', fontsize=14)
        
        # Subplot 3: MAE comparison
        train_mae = mean_absolute_error(train_true, train_pred)
        val_mae = mean_absolute_error(val_true, val_pred)
        test_mae = mean_absolute_error(test_true, test_pred)
        mae_values = [train_mae, val_mae, test_mae]
        
        bars = ax3.bar(datasets, mae_values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Dataset', fontsize=16)
        ax3.set_ylabel('MAE', fontsize=16)
        ax3.set_title('MAE Comparison', fontsize=18, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mae_val in zip(bars, mae_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mae_val:.4f}', ha='center', va='bottom', fontsize=14)
        
        # Subplot 4: Histogram of errors for all datasets
        ax4.hist(train_errors, bins=30, alpha=0.5, color='blue', label='Training', density=True)
        ax4.hist(val_errors, bins=30, alpha=0.5, color='orange', label='Validation', density=True)
        ax4.hist(test_errors, bins=30, alpha=0.5, color='green', label='Test', density=True)
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        ax4.set_xlabel('Prediction Error', fontsize=16)
        ax4.set_ylabel('Density', fontsize=16)
        ax4.set_title('Error Distribution Comparison', fontsize=18, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Performance Metrics Comparison: {name_str}', fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'performance_comparison_{name_str}.png'), dpi=400, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Performance comparison plot saved: performance_comparison_{name_str}.png")
        
        # 4. Create a comprehensive CSV with all metrics
        metrics_df = pd.DataFrame({
            'Dataset': ['Training', 'Validation', 'Test'],
            'Samples': [len(train_true), len(val_true), len(test_true)],
            'R²': [train_r2, val_r2, test_r2],
            'RMSE': [train_rmse, val_rmse, test_rmse],
            'MAE': [train_mae, val_mae, test_mae],
            'Error_Mean': [train_errors.mean(), val_errors.mean(), test_errors.mean()],
            'Error_Std': [train_errors.std(), val_errors.std(), test_errors.std()],
            'Error_Min': [train_errors.min(), val_errors.min(), test_errors.min()],
            'Error_Max': [train_errors.max(), val_errors.max(), test_errors.max()]
        })
        
        metrics_path = os.path.join(plot_dir, f'metrics_summary_{name_str}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  ✓ Metrics summary saved: metrics_summary_{name_str}.csv")


def generate_training_visualizations(history, save_dir):
    """Generate training process visualizations - each chart saved separately"""
    
    epochs = history['epoch']
    plot_dir = os.path.join(save_dir, 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Training and Validation Loss Curves - separate chart
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=3, alpha=0.8)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=3, alpha=0.8)
    plt.title('Training and Validation Loss Curves', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Loss curves saved: loss_curves.png")
    
    # 2. Training and Validation R² Scores Curves - separate chart
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['train_r2'], 'b-', label='Training R²', linewidth=3, alpha=0.8)
    plt.plot(epochs, history['val_r2'], 'r-', label='Validation R²', linewidth=3, alpha=0.8)
    plt.title('Training and Validation R² Scores', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('R² Score', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'r2_scores.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ R² scores saved: r2_scores.png")
    
    # 3. Learning Rate Schedule - separate chart
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['learning_rate'], 'g-', linewidth=3, alpha=0.8)
    plt.title('Learning Rate Schedule', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Learning Rate', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'learning_rate.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Learning rate schedule saved: learning_rate.png")
    
    # 4. Gradient Norm Change Curve - separate chart
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['grad_norm'], 'm-', linewidth=3, alpha=0.8)
    plt.title('Gradient Norm During Training', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Gradient Norm', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'gradient_norm.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Gradient norm saved: gradient_norm.png")
    
    # 5. Loss Difference Plot (overfitting indicator) - separate chart
    plt.figure(figsize=(12, 8))
    loss_diff = [val - train for train, val in zip(history['train_loss'], history['val_loss'])]
    plt.plot(epochs, loss_diff, 'orange', linewidth=3, alpha=0.8)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    plt.title('Overfitting Indicator (Val Loss - Train Loss)', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss Difference', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'overfitting_indicator.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Overfitting indicator saved: overfitting_indicator.png")
    
    # 6. Training Time per Epoch - separate chart
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, history['time_elapsed'], 'c-', linewidth=3, alpha=0.8)
    plt.title('Training Time per Epoch', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Time (seconds)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'time_per_epoch.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training time per epoch saved: time_per_epoch.png")
    
    # 7. Cumulative Training Time - separate chart
    plt.figure(figsize=(12, 8))
    cumulative_time = np.cumsum(history['time_elapsed'])
    plt.plot(epochs, cumulative_time / 60, 'purple', linewidth=3, alpha=0.8)  # Convert to minutes
    plt.title('Cumulative Training Time', fontsize=22, fontweight='bold')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Time (minutes)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'cumulative_time.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Cumulative training time saved: cumulative_time.png")
    
    # 8. Scatter plot of Training Loss vs Validation Loss - separate chart
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(history['train_loss'], history['val_loss'], 
                         c=history['val_r2'], cmap='viridis', alpha=0.7, s=80)
    plt.title('Train Loss vs Validation Loss (Colored by Val R²)', fontsize=22, fontweight='bold')
    plt.xlabel('Training Loss', fontsize=18)
    plt.ylabel('Validation Loss', fontsize=18)
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Validation R²', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'loss_scatter.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Loss scatter plot saved: loss_scatter.png")
    
    # 9. Training Summary - separate chart
    best_epoch = np.argmax(history['val_r2'])
    best_val_r2 = history['val_r2'][best_epoch]
    best_val_loss = history['val_loss'][best_epoch]
    total_training_time = np.sum(history['time_elapsed']) / 60  # minutes
    
    summary_text = f"Training Summary\n\n"
    summary_text += f"Best Model at Epoch: {best_epoch+1}\n"
    summary_text += f"Best Validation R²: {best_val_r2:.6f}\n"
    summary_text += f"Best Validation Loss: {best_val_loss:.6f}\n\n"
    summary_text += f"Final Training R²: {history['train_r2'][-1]:.6f}\n"
    summary_text += f"Final Validation R²: {history['val_r2'][-1]:.6f}\n\n"
    summary_text += f"Total Training Epochs: {len(epochs)}\n"
    summary_text += f"Total Training Time: {total_training_time:.1f} minutes\n"
    summary_text += f"Average Time per Epoch: {np.mean(history['time_elapsed']):.2f} seconds"
    
    plt.figure(figsize=(10, 8))
    plt.text(0.5, 0.5, summary_text, fontsize=18, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.title('Training Summary', fontsize=24, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_summary.png'), dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training summary saved: training_summary.png")


# ==================== 5. EXECUTION ====================
if __name__ == "__main__":
    print(f"\n{'='*100}")
    print(f"HIGH ACCURACY REGRESSION PIPELINE")
    print(f"{'='*100}")
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clean CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache to free memory")
    
    try:
        model, trainer, test_r2 = run_optimized_pipeline()
        
        print(f"\n{'='*100}")
        print(f"PIPELINE COMPLETED!")
        print(f"{'='*100}")
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_r2 = test_r2.get('overall', 0)
        print(f"\nFinal Test R²: {overall_r2:.6f}")
        
        if overall_r2 >= 0.96:
            print("🎉🎉🎉 EXCELLENT RESULTS! 🎉🎉🎉")
        elif overall_r2 >= 0.94:
            print("✅ Very good results!")
        elif overall_r2 >= 0.92:
            print("⚠️  Good results, but room for improvement")
        else:
            print("📈 Training completed. Consider:")
            print("  1. Increasing model size (carefully)")
            print("  2. Adding more training data")
            print("  3. Trying different feature engineering")
            print("  4. Longer training with more patience")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Suggested actions if CUDA runs out of memory
        if "out of memory" in str(e):
            print("\n💡 Tips to solve CUDA out of memory error:")
            print("  1. Reduce batch size further (try 32 or 16)")
            print("  2. Reduce model size (base_dim and num_blocks)")
            print("  3. Use gradient accumulation with smaller batches")
            print("  4. Disable ensemble and use single model")
            print("  5. Use CPU if GPU memory is too limited")
            print("  6. Reduce feature engineering complexity")