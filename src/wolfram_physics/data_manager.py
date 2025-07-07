"""
Data management module for Wolfram Physics Project.
Handles efficient storage, retrieval, and management of hypergraph evolution data.
"""

import json
import pickle
import zarr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import gzip
import sqlite3
from datetime import datetime
import hashlib
from .hypergraph_processor import HypergraphProcessor


class DataManager:
    """
    Manages storage and retrieval of hypergraph evolution data.
    Supports multiple storage backends and compression options.
    """
    
    def __init__(self, storage_path: Union[str, Path], 
                 backend: str = 'zarr', compress: bool = True):
        """
        Initialize the data manager.
        
        Args:
            storage_path: Path to storage directory
            backend: Storage backend ('zarr', 'json', 'pickle', 'sqlite')
            compress: Whether to use compression
        """
        self.storage_path = Path(storage_path)
        self.backend = backend
        self.compress = compress
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend-specific storage
        if backend == 'zarr':
            self.zarr_store = zarr.open(str(self.storage_path / 'hypergraph_data.zarr'), mode='a')
        elif backend == 'sqlite':
            self.db_path = self.storage_path / 'hypergraph_data.db'
            self._init_sqlite_db()
        
        self.metadata = self._load_metadata()
    
    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parameters TEXT,
                    hash TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    step INTEGER,
                    nodes INTEGER,
                    edges INTEGER,
                    data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evolution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    step INTEGER,
                    rule_name TEXT,
                    application_count INTEGER,
                    log_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')
            
            conn.commit()
    
    def _load_metadata(self) -> Dict:
        """Load metadata about stored experiments."""
        metadata_path = self.storage_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {'experiments': {}, 'created_at': datetime.now().isoformat()}
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        metadata_path = self.storage_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash for data integrity."""
        data_str = json.dumps(data, sort_keys=True) if isinstance(data, dict) else str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression."""
        if self.backend == 'json':
            serialized = json.dumps(data).encode()
        else:
            serialized = pickle.dumps(data)
        
        if self.compress:
            serialized = gzip.compress(serialized)
        
        return serialized
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with optional decompression."""
        if self.compress:
            data = gzip.decompress(data)
        
        if self.backend == 'json':
            return json.loads(data.decode())
        else:
            return pickle.loads(data)
    
    def save_experiment(self, experiment_name: str, processor: HypergraphProcessor,
                       description: str = "", parameters: Optional[Dict] = None) -> str:
        """
        Save an experiment with its hypergraph state.
        
        Args:
            experiment_name: Name of the experiment
            processor: HypergraphProcessor to save
            description: Optional description
            parameters: Optional parameters dictionary
            
        Returns:
            Experiment ID
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare experiment data
        experiment_data = {
            'name': experiment_name,
            'description': description,
            'parameters': parameters or {},
            'snapshot': processor.snapshot(),
            'evolution_history': processor.evolution_history,
            'created_at': datetime.now().isoformat()
        }
        
        # Compute hash
        data_hash = self._compute_hash(experiment_data)
        experiment_data['hash'] = data_hash
        
        if self.backend == 'zarr':
            self._save_to_zarr(experiment_id, experiment_data)
        elif self.backend == 'sqlite':
            self._save_to_sqlite(experiment_id, experiment_data)
        else:
            self._save_to_file(experiment_id, experiment_data)
        
        # Update metadata
        self.metadata['experiments'][experiment_id] = {
            'name': experiment_name,
            'description': description,
            'created_at': experiment_data['created_at'],
            'hash': data_hash,
            'node_count': processor.node_count,
            'edge_count': processor.edge_count,
            'evolution_steps': len(processor.evolution_history)
        }
        self._save_metadata()
        
        return experiment_id
    
    def _save_to_zarr(self, experiment_id: str, data: Dict) -> None:
        """Save data using Zarr backend."""
        group = self.zarr_store.create_group(experiment_id, overwrite=True)
        
        # Save metadata
        group.attrs['name'] = data['name']
        group.attrs['description'] = data['description']
        group.attrs['created_at'] = data['created_at']
        group.attrs['hash'] = data['hash']
        group.attrs['parameters'] = json.dumps(data['parameters'])
        
        # Save current snapshot
        snapshot = data['snapshot']
        group.create_dataset('current_nodes', data=list(snapshot['nodes']))
        group.create_dataset('current_edges', data=json.dumps(snapshot['edges']))
        
        # Save evolution history
        if data['evolution_history']:
            history_group = group.create_group('evolution_history')
            for i, hist_snapshot in enumerate(data['evolution_history']):
                step_group = history_group.create_group(f'step_{i}')
                step_group.attrs['step'] = hist_snapshot['step']
                step_group.attrs['node_count'] = hist_snapshot['node_count']
                step_group.attrs['edge_count'] = hist_snapshot['edge_count']
                step_group.create_dataset('nodes', data=list(hist_snapshot['nodes']))
                step_group.create_dataset('edges', data=json.dumps(hist_snapshot['edges']))
    
    def _save_to_sqlite(self, experiment_id: str, data: Dict) -> None:
        """Save data using SQLite backend."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert experiment
            cursor.execute('''
                INSERT OR REPLACE INTO experiments (name, description, created_at, parameters, hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (data['name'], data['description'], data['created_at'], 
                  json.dumps(data['parameters']), data['hash']))
            
            experiment_pk = cursor.lastrowid
            
            # Insert current snapshot
            snapshot_data = self._serialize_data(data['snapshot'])
            cursor.execute('''
                INSERT INTO snapshots (experiment_id, step, nodes, edges, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (experiment_pk, data['snapshot']['step'], 
                  data['snapshot']['node_count'], data['snapshot']['edge_count'], 
                  snapshot_data))
            
            # Insert evolution history
            for hist_snapshot in data['evolution_history']:
                hist_data = self._serialize_data(hist_snapshot)
                cursor.execute('''
                    INSERT INTO snapshots (experiment_id, step, nodes, edges, data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (experiment_pk, hist_snapshot['step'], 
                      hist_snapshot['node_count'], hist_snapshot['edge_count'], 
                      hist_data))
            
            conn.commit()
    
    def _save_to_file(self, experiment_id: str, data: Dict) -> None:
        """Save data to file system."""
        experiment_path = self.storage_path / f"{experiment_id}.{self.backend}"
        
        if self.backend == 'json':
            with open(experiment_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:  # pickle
            with open(experiment_path, 'wb') as f:
                pickle.dump(data, f)
    
    def load_experiment(self, experiment_id: str) -> Optional[HypergraphProcessor]:
        """
        Load an experiment and return a HypergraphProcessor.
        
        Args:
            experiment_id: ID of the experiment to load
            
        Returns:
            HypergraphProcessor with loaded state or None if not found
        """
        if self.backend == 'zarr':
            return self._load_from_zarr(experiment_id)
        elif self.backend == 'sqlite':
            return self._load_from_sqlite(experiment_id)
        else:
            return self._load_from_file(experiment_id)
    
    def _load_from_zarr(self, experiment_id: str) -> Optional[HypergraphProcessor]:
        """Load data from Zarr backend."""
        if experiment_id not in self.zarr_store:
            return None
        
        group = self.zarr_store[experiment_id]
        
        # Load current snapshot
        edges_data = json.loads(group['current_edges'][()])
        processor = HypergraphProcessor(edges_data)
        
        # Load evolution history
        if 'evolution_history' in group:
            history_group = group['evolution_history']
            for step_key in sorted(history_group.keys()):
                step_group = history_group[step_key]
                snapshot = {
                    'step': step_group.attrs['step'],
                    'node_count': step_group.attrs['node_count'],
                    'edge_count': step_group.attrs['edge_count'],
                    'nodes': list(step_group['nodes']),
                    'edges': json.loads(step_group['edges'][()])
                }
                processor.evolution_history.append(snapshot)
        
        return processor
    
    def _load_from_sqlite(self, experiment_name: str) -> Optional[HypergraphProcessor]:
        """Load data from SQLite backend."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find experiment
            cursor.execute('SELECT id FROM experiments WHERE name = ?', (experiment_name,))
            result = cursor.fetchone()
            if not result:
                return None
            
            experiment_id = result[0]
            
            # Load latest snapshot
            cursor.execute('''
                SELECT data FROM snapshots 
                WHERE experiment_id = ? 
                ORDER BY step DESC LIMIT 1
            ''', (experiment_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            snapshot_data = self._deserialize_data(result[0])
            processor = HypergraphProcessor(snapshot_data['edges'])
            
            # Load evolution history
            cursor.execute('''
                SELECT data FROM snapshots 
                WHERE experiment_id = ? 
                ORDER BY step ASC
            ''', (experiment_id,))
            
            for row in cursor.fetchall():
                hist_data = self._deserialize_data(row[0])
                processor.evolution_history.append(hist_data)
            
            return processor
    
    def _load_from_file(self, experiment_id: str) -> Optional[HypergraphProcessor]:
        """Load data from file system."""
        experiment_path = self.storage_path / f"{experiment_id}.{self.backend}"
        
        if not experiment_path.exists():
            return None
        
        if self.backend == 'json':
            with open(experiment_path, 'r') as f:
                data = json.load(f)
        else:  # pickle
            with open(experiment_path, 'rb') as f:
                data = pickle.load(f)
        
        # Create processor from snapshot
        processor = HypergraphProcessor(data['snapshot']['edges'])
        processor.evolution_history = data['evolution_history']
        processor.current_step = data['snapshot']['step']
        
        return processor
    
    def list_experiments(self) -> List[Dict]:
        """
        List all stored experiments.
        
        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []
        for exp_id, exp_data in self.metadata['experiments'].items():
            experiments.append({
                'id': exp_id,
                'name': exp_data['name'],
                'description': exp_data['description'],
                'created_at': exp_data['created_at'],
                'node_count': exp_data['node_count'],
                'edge_count': exp_data['edge_count'],
                'evolution_steps': exp_data['evolution_steps']
            })
        
        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment from storage.
        
        Args:
            experiment_id: ID of the experiment to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if experiment_id not in self.metadata['experiments']:
            return False
        
        try:
            if self.backend == 'zarr':
                if experiment_id in self.zarr_store:
                    del self.zarr_store[experiment_id]
            elif self.backend == 'sqlite':
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM experiments WHERE name = ?', (experiment_id,))
                    conn.commit()
            else:
                experiment_path = self.storage_path / f"{experiment_id}.{self.backend}"
                if experiment_path.exists():
                    experiment_path.unlink()
            
            # Remove from metadata
            del self.metadata['experiments'][experiment_id]
            self._save_metadata()
            
            return True
        except Exception:
            return False
    
    def get_storage_info(self) -> Dict:
        """
        Get information about storage usage.
        
        Returns:
            Dictionary with storage statistics
        """
        info = {
            'backend': self.backend,
            'storage_path': str(self.storage_path),
            'compression': self.compress,
            'experiment_count': len(self.metadata['experiments']),
            'total_size_bytes': 0
        }
        
        # Calculate total size
        for file_path in self.storage_path.rglob('*'):
            if file_path.is_file():
                info['total_size_bytes'] += file_path.stat().st_size
        
        info['total_size_mb'] = info['total_size_bytes'] / (1024 * 1024)
        
        return info
    
    def export_experiment(self, experiment_id: str, export_path: Union[str, Path], 
                         format: str = 'json') -> bool:
        """
        Export an experiment to a specific format.
        
        Args:
            experiment_id: ID of the experiment to export
            export_path: Path to export to
            format: Export format ('json', 'pickle')
            
        Returns:
            True if export was successful, False otherwise
        """
        processor = self.load_experiment(experiment_id)
        if not processor:
            return False
        
        export_data = {
            'experiment_id': experiment_id,
            'metadata': self.metadata['experiments'][experiment_id],
            'snapshot': processor.snapshot(),
            'evolution_history': processor.evolution_history,
            'exported_at': datetime.now().isoformat()
        }
        
        export_path = Path(export_path)
        
        try:
            if format == 'json':
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:  # pickle
                with open(export_path, 'wb') as f:
                    pickle.dump(export_data, f)
            
            return True
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of the data manager."""
        return f"DataManager(backend={self.backend}, experiments={len(self.metadata['experiments'])})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()