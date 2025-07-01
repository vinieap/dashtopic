#!/usr/bin/env python3
"""
API Explorer - Discover your actual service APIs

This script helps you understand what methods and properties 
are actually available in your services, so you can write 
accurate tests.

Usage:
    python explore_api.py                    # Explore all services
    python explore_api.py FileIOService     # Explore specific service
    python explore_api.py DataConfig        # Explore data models
"""

import sys
import inspect
from typing import Any, Dict, List


def explore_class(cls: type, class_name: str) -> Dict[str, Any]:
    """Explore a class and return information about its methods and properties."""
    info = {
        'class_name': class_name,
        'methods': [],
        'properties': [],
        'attributes': [],
        'constructor_signature': None
    }
    
    # Get constructor signature
    try:
        sig = inspect.signature(cls.__init__)
        info['constructor_signature'] = str(sig)
    except:
        info['constructor_signature'] = "Could not determine"
    
    # Get all members
    for name, member in inspect.getmembers(cls):
        if name.startswith('_'):
            continue
            
        if inspect.ismethod(member) or inspect.isfunction(member):
            try:
                sig = inspect.signature(member)
                info['methods'].append({
                    'name': name,
                    'signature': f"{name}{sig}",
                    'docstring': inspect.getdoc(member) or "No docstring"
                })
            except:
                info['methods'].append({
                    'name': name,
                    'signature': f"{name}(...)",
                    'docstring': "Could not determine signature"
                })
        elif isinstance(member, property):
            info['properties'].append({
                'name': name,
                'docstring': inspect.getdoc(member) or "No docstring"
            })
        else:
            # Regular attributes
            info['attributes'].append({
                'name': name,
                'type': type(member).__name__,
                'value': str(member) if len(str(member)) < 100 else "Large value..."
            })
    
    return info


def print_class_info(info: Dict[str, Any]):
    """Print formatted information about a class."""
    print(f"\n{'='*60}")
    print(f"🔍 {info['class_name']}")
    print(f"{'='*60}")
    
    print(f"\n📝 Constructor:")
    print(f"   {info['class_name']}{info['constructor_signature']}")
    
    if info['methods']:
        print(f"\n🛠️  Methods ({len(info['methods'])}):")
        for method in sorted(info['methods'], key=lambda x: x['name']):
            print(f"   • {method['signature']}")
            if method['docstring'] and method['docstring'] != "No docstring":
                # Show first line of docstring
                first_line = method['docstring'].split('\n')[0]
                print(f"     └─ {first_line}")
    
    if info['properties']:
        print(f"\n🏷️  Properties ({len(info['properties'])}):")
        for prop in sorted(info['properties'], key=lambda x: x['name']):
            print(f"   • {prop['name']}")
            if prop['docstring'] and prop['docstring'] != "No docstring":
                first_line = prop['docstring'].split('\n')[0]
                print(f"     └─ {first_line}")
    
    if info['attributes']:
        print(f"\n📦 Attributes ({len(info['attributes'])}):")
        for attr in sorted(info['attributes'], key=lambda x: x['name'])[:10]:  # Limit to first 10
            print(f"   • {attr['name']}: {attr['type']}")


def explore_services():
    """Explore all available services."""
    services = [
        ('FileIOService', 'src.services.file_io_service'),
        ('CacheService', 'src.services.cache_service'),
        ('DataValidationService', 'src.services.data_validation_service'),
        ('EmbeddingService', 'src.services.embedding_service'),
        ('BERTopicService', 'src.services.bertopic_service'),
    ]
    
    for service_name, module_path in services:
        try:
            module = __import__(module_path, fromlist=[service_name])
            service_class = getattr(module, service_name)
            info = explore_class(service_class, service_name)
            print_class_info(info)
        except Exception as e:
            print(f"\n❌ Could not explore {service_name}: {e}")


def explore_models():
    """Explore data models."""
    models = [
        ('FileMetadata', 'src.models.data_models'),
        ('DataConfig', 'src.models.data_models'),
        ('ModelInfo', 'src.models.data_models'),
        ('CacheInfo', 'src.models.data_models'),
    ]
    
    for model_name, module_path in models:
        try:
            module = __import__(module_path, fromlist=[model_name])
            model_class = getattr(module, model_name)
            info = explore_class(model_class, model_name)
            print_class_info(info)
        except Exception as e:
            print(f"\n❌ Could not explore {model_name}: {e}")


def explore_specific(class_name: str):
    """Explore a specific class."""
    # Try to find it in services
    service_modules = [
        'src.services.file_io_service',
        'src.services.cache_service', 
        'src.services.data_validation_service',
        'src.services.embedding_service',
        'src.services.bertopic_service',
    ]
    
    # Try to find it in models
    model_modules = [
        'src.models.data_models',
        'src.models.optimization_models',
    ]
    
    all_modules = service_modules + model_modules
    
    for module_path in all_modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                target_class = getattr(module, class_name)
                info = explore_class(target_class, class_name)
                print_class_info(info)
                return
        except:
            continue
    
    print(f"❌ Could not find class '{class_name}' in any known module")


def create_test_template(class_name: str):
    """Create a test template for a specific class."""
    print(f"\n🧪 Test Template for {class_name}:")
    print(f"{'='*40}")
    
    template = f'''
def test_{class_name.lower()}_initialization(self):
    """Test that {class_name} can be created."""
    from src.services.{class_name.lower()}_service import {class_name}  # Adjust import
    
    service = {class_name}()
    assert service is not None

def test_{class_name.lower()}_basic_functionality(self):
    """Test basic {class_name} functionality."""
    from src.services.{class_name.lower()}_service import {class_name}  # Adjust import
    
    service = {class_name}()
    
    # Add your specific tests here
    # result = service.some_method()
    # assert result == expected_value
'''
    
    print(template)


def main():
    """Main entry point."""
    print("🔍 BERTopic Desktop Application - API Explorer")
    print("=" * 50)
    
    if len(sys.argv) == 1:
        # No arguments - explore everything
        print("\n🛠️ Exploring Services...")
        explore_services()
        
        print("\n📊 Exploring Data Models...")
        explore_models()
        
        print("\n💡 Usage Tips:")
        print("   python explore_api.py FileIOService     # Explore specific service")
        print("   python explore_api.py DataConfig        # Explore specific model")
        
    elif len(sys.argv) == 2:
        class_name = sys.argv[1]
        
        if class_name.lower() == 'services':
            explore_services()
        elif class_name.lower() == 'models':
            explore_models()
        else:
            explore_specific(class_name)
            create_test_template(class_name)
    
    else:
        print("Usage: python explore_api.py [ClassName]")
        sys.exit(1)


if __name__ == "__main__":
    main()