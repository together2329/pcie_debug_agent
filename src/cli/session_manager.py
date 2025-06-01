"""
Session management for PCIe Debug Agent
Handles conversation persistence and resumption
"""

import json
import uuid
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class SessionManager:
    """Manages conversation sessions"""
    
    def __init__(self, sessions_dir: Optional[Path] = None):
        self.sessions_dir = sessions_dir or Path.home() / ".pcie_debug" / "sessions"
        self.sessions_index = self.sessions_dir / "index.json"
        
        # Create directory if it doesn't exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Load sessions index
        self._index = self._load_index()
    
    def _load_index(self) -> List[Dict[str, Any]]:
        """Load sessions index"""
        if self.sessions_index.exists():
            try:
                with open(self.sessions_index, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def _save_index(self):
        """Save sessions index"""
        try:
            # Sort by timestamp (newest first)
            self._index.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            with open(self.sessions_index, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            print(f"Failed to save sessions index: {e}")
    
    def save_session(self, conversation: List[Dict[str, Any]], 
                    title: Optional[str] = None, 
                    tags: Optional[List[str]] = None) -> str:
        """Save a conversation session"""
        
        session_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Generate title if not provided
        if not title:
            if conversation:
                first_message = next((msg for msg in conversation if msg.get('role') == 'user'), None)
                if first_message:
                    title = first_message.get('content', '')[:50] + "..."
                else:
                    title = f"PCIe Debug Session"
            else:
                title = f"Empty Session"
        
        # Create session data
        session_data = {
            "id": session_id,
            "title": title,
            "timestamp": timestamp,
            "created": datetime.now().isoformat(),
            "conversation": conversation,
            "tags": tags or [],
            "turn_count": len([msg for msg in conversation if msg.get('role') == 'user']),
            "last_model": self._extract_model_from_conversation(conversation)
        }
        
        # Save session file
        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save session: {e}")
            return None
        
        # Add to index
        index_entry = {
            "id": session_id,
            "title": title,
            "timestamp": timestamp,
            "created": datetime.now().isoformat(),
            "turn_count": session_data["turn_count"],
            "last_model": session_data["last_model"],
            "tags": tags or []
        }
        
        self._index.append(index_entry)
        self._save_index()
        
        return session_id
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session by ID"""
        # Handle partial IDs (first 8 characters)
        if len(session_id) == 8:
            matching_sessions = [s for s in self._index if s['id'].startswith(session_id)]
            if len(matching_sessions) == 1:
                session_id = matching_sessions[0]['id']
            elif len(matching_sessions) > 1:
                print(f"Multiple sessions match '{session_id}'. Please use more characters.")
                return None
            else:
                print(f"No session found matching '{session_id}'")
                return None
        
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def list_sessions(self, limit: int = 20, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent sessions"""
        sessions = self._index.copy()
        
        # Filter by tag if specified
        if tag:
            sessions = [s for s in sessions if tag in s.get('tags', [])]
        
        # Return most recent sessions
        return sessions[:limit]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        # Handle partial IDs
        if len(session_id) == 8:
            matching_sessions = [s for s in self._index if s['id'].startswith(session_id)]
            if len(matching_sessions) == 1:
                session_id = matching_sessions[0]['id']
            elif len(matching_sessions) > 1:
                print(f"Multiple sessions match '{session_id}'. Please use more characters.")
                return False
            else:
                return False
        
        # Remove from index
        self._index = [s for s in self._index if s['id'] != session_id]
        self._save_index()
        
        # Remove session file
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            try:
                session_file.unlink()
                return True
            except Exception:
                return False
        
        return True
    
    def get_most_recent_session(self) -> Optional[Dict[str, Any]]:
        """Get the most recent session"""
        if self._index:
            return self.load_session(self._index[0]['id'])
        return None
    
    def search_sessions(self, query: str) -> List[Dict[str, Any]]:
        """Search sessions by title or content"""
        query_lower = query.lower()
        matching_sessions = []
        
        for session_info in self._index:
            # Check title
            if query_lower in session_info.get('title', '').lower():
                matching_sessions.append(session_info)
                continue
            
            # Check conversation content
            session = self.load_session(session_info['id'])
            if session:
                for message in session.get('conversation', []):
                    content = message.get('content', '')
                    if query_lower in content.lower():
                        matching_sessions.append(session_info)
                        break
        
        return matching_sessions
    
    def add_tags_to_session(self, session_id: str, tags: List[str]) -> bool:
        """Add tags to a session"""
        session = self.load_session(session_id)
        if not session:
            return False
        
        # Update session file
        existing_tags = set(session.get('tags', []))
        existing_tags.update(tags)
        session['tags'] = list(existing_tags)
        
        session_file = self.sessions_dir / f"{session_id}.json"
        try:
            with open(session_file, 'w') as f:
                json.dump(session, f, indent=2)
        except Exception:
            return False
        
        # Update index
        for idx, session_info in enumerate(self._index):
            if session_info['id'] == session_id:
                self._index[idx]['tags'] = session['tags']
                break
        
        self._save_index()
        return True
    
    def _extract_model_from_conversation(self, conversation: List[Dict[str, Any]]) -> str:
        """Extract model information from conversation"""
        # Look for model information in conversation
        for message in reversed(conversation):
            if message.get('role') == 'system' and 'model' in message.get('content', '').lower():
                # Try to extract model name
                content = message.get('content', '')
                if 'llama' in content.lower():
                    return 'llama-3.2-3b'
                elif 'deepseek' in content.lower():
                    return 'deepseek-r1-7b'
                elif 'gpt-4' in content.lower():
                    return 'gpt-4'
                elif 'claude' in content.lower():
                    return 'claude-3-opus'
        
        return 'unknown'
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about sessions"""
        total_sessions = len(self._index)
        total_turns = sum(s.get('turn_count', 0) for s in self._index)
        
        # Model usage stats
        model_usage = {}
        for session in self._index:
            model = session.get('last_model', 'unknown')
            model_usage[model] = model_usage.get(model, 0) + 1
        
        # Recent activity (last 7 days)
        recent_threshold = time.time() - (7 * 24 * 60 * 60)
        recent_sessions = [s for s in self._index if s.get('timestamp', 0) > recent_threshold]
        
        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "recent_sessions": len(recent_sessions),
            "model_usage": model_usage,
            "avg_turns_per_session": total_turns / total_sessions if total_sessions > 0 else 0
        }