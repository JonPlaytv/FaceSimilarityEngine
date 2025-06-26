#!/usr/bin/env python3
"""
Create sample face images using generated SVG portraits for initial dataset
"""

import os
import requests
from PIL import Image, ImageDraw, ImageFont
import io

def create_sample_face_svg(face_id, name, features):
    """Create a simple SVG face portrait"""
    svg_content = f'''
    <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
        <!-- Background -->
        <rect width="200" height="200" fill="{features['bg_color']}"/>
        
        <!-- Face -->
        <ellipse cx="100" cy="110" rx="60" ry="70" fill="{features['skin_color']}" stroke="#333" stroke-width="2"/>
        
        <!-- Eyes -->
        <ellipse cx="85" cy="95" rx="8" ry="6" fill="white"/>
        <ellipse cx="115" cy="95" rx="8" ry="6" fill="white"/>
        <circle cx="85" cy="95" r="4" fill="{features['eye_color']}"/>
        <circle cx="115" cy="95" r="4" fill="{features['eye_color']}"/>
        <circle cx="85" cy="93" r="2" fill="black"/>
        <circle cx="115" cy="93" r="2" fill="black"/>
        
        <!-- Eyebrows -->
        <path d="M 75 85 Q 85 80 95 85" stroke="black" stroke-width="3" fill="none"/>
        <path d="M 105 85 Q 115 80 125 85" stroke="black" stroke-width="3" fill="none"/>
        
        <!-- Nose -->
        <path d="M 100 105 L 95 115 L 100 118 L 105 115 Z" fill="{features['skin_color']}" stroke="#666" stroke-width="1"/>
        
        <!-- Mouth -->
        <path d="M 90 130 Q 100 {features['mouth_curve']} 110 130" stroke="black" stroke-width="2" fill="none"/>
        
        <!-- Hair -->
        <path d="M 40 70 Q 100 {features['hair_height']} 160 70 Q 160 90 150 100 L 50 100 Q 40 90 40 70" 
              fill="{features['hair_color']}" stroke="#333" stroke-width="1"/>
        
        <!-- Text label -->
        <text x="100" y="190" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="black">
            {name}
        </text>
    </svg>
    '''
    return svg_content

def svg_to_image(svg_content, output_path):
    """Convert SVG to PNG image"""
    try:
        # Try using cairosvg if available, otherwise create a simple placeholder
        try:
            import cairosvg
            png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))
            with open(output_path, 'wb') as f:
                f.write(png_data)
            return True
        except ImportError:
            # Fallback: create a simple colored rectangle as placeholder
            img = Image.new('RGB', (200, 200), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # Draw a simple face
            draw.ellipse([50, 60, 150, 160], fill='peachpuff', outline='black', width=2)
            draw.ellipse([70, 90, 80, 100], fill='white', outline='black')
            draw.ellipse([120, 90, 130, 100], fill='white', outline='black')
            draw.ellipse([73, 93, 77, 97], fill='blue')
            draw.ellipse([123, 93, 127, 97], fill='blue')
            draw.arc([90, 110, 110, 130], 0, 180, fill='red', width=2)
            
            img.save(output_path, 'PNG')
            return True
    except Exception as e:
        print(f"Error creating image {output_path}: {e}")
        return False

def create_sample_dataset():
    """Create sample face images for the dataset"""
    dataset_dir = "static/dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Sample face configurations
    faces = [
        {
            'id': 1,
            'name': 'Sample Face 1',
            'features': {
                'bg_color': '#f0f8ff',
                'skin_color': '#fdbcb4',
                'eye_color': '#4169e1',
                'hair_color': '#8b4513',
                'hair_height': 40,
                'mouth_curve': 135
            }
        },
        {
            'id': 2,
            'name': 'Sample Face 2',
            'features': {
                'bg_color': '#f5f5dc',
                'skin_color': '#deb887',
                'eye_color': '#228b22',
                'hair_color': '#000000',
                'hair_height': 35,
                'mouth_curve': 140
            }
        },
        {
            'id': 3,
            'name': 'Sample Face 3',
            'features': {
                'bg_color': '#ffe4e1',
                'skin_color': '#f5deb3',
                'eye_color': '#8b4513',
                'hair_color': '#ffd700',
                'hair_height': 45,
                'mouth_curve': 138
            }
        },
        {
            'id': 4,
            'name': 'Sample Face 4',
            'features': {
                'bg_color': '#e6e6fa',
                'skin_color': '#cd853f',
                'eye_color': '#2f4f4f',
                'hair_color': '#696969',
                'hair_height': 38,
                'mouth_curve': 142
            }
        },
        {
            'id': 5,
            'name': 'Sample Face 5',
            'features': {
                'bg_color': '#f0fff0',
                'skin_color': '#f0e68c',
                'eye_color': '#dc143c',
                'hair_color': '#ff6347',
                'hair_height': 42,
                'mouth_curve': 136
            }
        }
    ]
    
    created_count = 0
    for face in faces:
        svg_content = create_sample_face_svg(face['id'], face['name'], face['features'])
        output_path = os.path.join(dataset_dir, f"sample_face_{face['id']}.png")
        
        if svg_to_image(svg_content, output_path):
            created_count += 1
            print(f"Created: {output_path}")
        else:
            print(f"Failed to create: {output_path}")
    
    print(f"Successfully created {created_count} sample face images in {dataset_dir}")
    return created_count

if __name__ == "__main__":
    create_sample_dataset()