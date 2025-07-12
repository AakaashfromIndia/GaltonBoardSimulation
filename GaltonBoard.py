import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys
import math

class GaltonBoardSimulation:
    def __init__(self):
        pygame.init()
        
        # 1200x700 layout
        self.width = 1200
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Galton Board Simulation')
        
        # Professional color scheme
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (70, 130, 180)
        self.RED = (220, 20, 60)
        self.GREEN = (34, 139, 34)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (240, 240, 240)
        self.DARK_BLUE = (25, 25, 112)
        self.ORANGE = (255, 140, 0)
        self.LIGHT_BLUE = (173, 216, 230)
        self.YELLOW = (255, 215, 0)
        self.DARK_GRAY = (64, 64, 64)
        self.SLIDER_TRACK = (200, 200, 200)
        self.SLIDER_HANDLE = (70, 130, 180)
        self.SLIDER_ACTIVE = (255, 140, 0)
        
        # Simulation parameters
        self.n_balls = 500
        self.n_rows = 12
        self.probability = 0.5
        self.ball_speed = 6
        self.max_simultaneous_balls = 8
        
        # Performance optimization variables
        self.plot_update_counter = 0
        self.plot_update_interval = 10
        
        # Multi-ball animation state
        self.ball_paths = []
        self.balls_in_flight = []
        self.balls_dropped = 0
        self.animation_delay = 5
        self.delay_counter = 0
        self.ball_falling = False
        
        # Results tracking
        self.positions = []
        self.bins_count = np.zeros(self.n_rows + 1, dtype=int)
        self.balls_completed = 0
        self.observed_mean = 0
        self.observed_std = 0
        self.theoretical_mean = 0
        self.theoretical_std = 0
        self.simulation_complete = False
        
        # UI elements
        self.title_font = pygame.font.Font(None, 32)
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.label_font = pygame.font.Font(None, 20)
        
        # Optimized layout with proper spacing
        self.header_height = 60
        
        # Board area with REDUCED safe zone
        self.board_width = 600
        self.board_height = 480
        self.board_margin = 20
        self.board_x = 300
        self.board_y = self.header_height + 40
        
        self.board_area = {'x': self.board_x, 'y': self.board_y, 'width': self.board_width, 'height': self.board_height}
        
        # Plot area
        self.plot_area = {'x': 920, 'y': self.header_height + 20, 'width': 260, 'height': 360}
        
        # Progress bar area
        self.progress_area = {'x': 920, 'y': self.header_height + 380, 'width': 260, 'height': 40}
        
        # Control panel layout
        self.slider_panel = {'x': 20, 'y': self.header_height + 20, 'width': 260, 'height': 520}
        
        # Professional slider configuration
        self.slider_width = 200
        self.slider_height = 20
        self.slider_spacing = 70
        self.slider_start_y = self.header_height + 60
        
        self.sliders = {
            'balls': {
                'x': 40, 'y': self.slider_start_y, 'width': self.slider_width, 'height': self.slider_height,
                'min': 100, 'max': 1000, 'value': self.n_balls,
                'label': 'Number of Balls', 'step': 50, 'unit': ''
            },
            'rows': {
                'x': 40, 'y': self.slider_start_y + self.slider_spacing, 'width': self.slider_width, 'height': self.slider_height,
                'min': 5, 'max': 18, 'value': self.n_rows,
                'label': 'Number of Rows', 'step': 1, 'unit': ''
            },
            'prob': {
                'x': 40, 'y': self.slider_start_y + 2 * self.slider_spacing, 'width': self.slider_width, 'height': self.slider_height,
                'min': 0.1, 'max': 0.9, 'value': self.probability,
                'label': 'Probability (Right)', 'step': 0.01, 'unit': ''
            },
            'speed': {
                'x': 40, 'y': self.slider_start_y + 3 * self.slider_spacing, 'width': self.slider_width, 'height': self.slider_height,
                'min': 1, 'max': 100, 'value': self.ball_speed,
                'label': 'Animation Speed', 'step': 1, 'unit': 'x'
            },
            'multi': {
                'x': 40, 'y': self.slider_start_y + 4 * self.slider_spacing, 'width': self.slider_width, 'height': self.slider_height,
                'min': 1, 'max': 25, 'value': self.max_simultaneous_balls,
                'label': 'Multi-Ball Count', 'step': 1, 'unit': ''
            }
        }
        
        # Professional buttons
        self.button_height = 40
        self.button_width = 110
        self.button_y = self.slider_start_y + 5 * self.slider_spacing + 20
        self.buttons = {
            'run': {'x': 40, 'y': self.button_y, 'width': self.button_width, 'height': self.button_height, 'text': 'START'},
            'pause': {'x': 40, 'y': self.button_y + 50, 'width': self.button_width, 'height': self.button_height, 'text': 'PAUSE'},
            'reset': {'x': 160, 'y': self.button_y, 'width': self.button_width, 'height': self.button_height, 'text': 'RESET'},
            'stop': {'x': 160, 'y': self.button_y + 50, 'width': self.button_width, 'height': self.button_height, 'text': 'STOP'}
        }
        
        # Performance optimization
        self.clock = pygame.time.Clock()
        self.running = True
        self.dragging = None
        self.paused = False
        self.stopped = False
        
        # Plot surface caching
        self.plot_surface = None
        self.plot_needs_update = False
        
        # Pre-calculated positions
        self.peg_positions = {}
        self.bin_positions = {}
        
        self.calculate_theoretical_values()
        self.precalculate_positions()
    
    def update_bins_array_size(self):
        """Update bins_count array size when n_rows changes"""
        new_size = self.n_rows + 1
        if len(self.bins_count) != new_size:
            old_bins = self.bins_count.copy()
            self.bins_count = np.zeros(new_size, dtype=int)
            copy_size = min(len(old_bins), new_size)
            self.bins_count[:copy_size] = old_bins[:copy_size]
    
    def precalculate_positions(self):
        """Pre-calculate positions with REDUCED safe zone"""
        self.update_bins_array_size()
        
        area = self.board_area
        
        # REDUCED safe zone - smaller margin
        safe_margin = self.board_margin  # Now 40 instead of 80
        usable_width = area['width'] - (2 * safe_margin)
        usable_height = area['height'] - (2 * safe_margin) - 80
        
        # Improved spacing calculations
        peg_spacing_x = usable_width / (self.n_rows + 1)
        peg_spacing_y = usable_height / (self.n_rows + 1)
        
        # Pre-calculate peg positions with reduced safe zone
        self.peg_positions = {}
        for row in range(self.n_rows):
            for col in range(row + 1):
                # Better row centering
                row_width = row * peg_spacing_x
                row_start_x = area['x'] + safe_margin + (usable_width - row_width) / 2
                x = row_start_x + col * peg_spacing_x
                y = area['y'] + safe_margin + row * peg_spacing_y + 40
                
                # Ensure positions are within bounds
                x = max(area['x'] + safe_margin, min(x, area['x'] + area['width'] - safe_margin))
                y = max(area['y'] + safe_margin, min(y, area['y'] + area['height'] - safe_margin - 80))
                
                self.peg_positions[(row, col)] = (int(x), int(y))
        
        # Pre-calculate bin positions with reduced safe zone
        self.bin_positions = {}
        bin_y = area['y'] + area['height'] - 60
        bin_row_width = self.n_rows * peg_spacing_x
        bin_row_start_x = area['x'] + safe_margin + (usable_width - bin_row_width) / 2
        
        for bin_num in range(self.n_rows + 1):
            x = bin_row_start_x + bin_num * peg_spacing_x
            x = max(area['x'] + safe_margin, min(x, area['x'] + area['width'] - safe_margin))
            self.bin_positions[bin_num] = (int(x), int(bin_y))
    
    def calculate_theoretical_values(self):
        """Calculate theoretical mean and standard deviation"""
        self.theoretical_mean = self.n_rows * self.probability
        self.theoretical_std = np.sqrt(self.n_rows * self.probability * (1 - self.probability))
    
    def generate_ball_paths(self):
        """Pre-generate random paths for all balls"""
        self.ball_paths = []
        for _ in range(self.n_balls):
            path = []
            position = 0
            for _ in range(self.n_rows):
                if np.random.random() < self.probability:
                    position += 1
                path.append(position)
            self.ball_paths.append(path)
    
    def start_animation(self):
        """Initialize animation"""
        self.update_bins_array_size()
        
        self.positions = []
        self.bins_count.fill(0)
        self.balls_completed = 0
        self.observed_mean = 0
        self.observed_std = 0
        self.simulation_complete = False
        self.balls_dropped = 0
        self.balls_in_flight = []
        self.ball_falling = True
        self.paused = False
        self.stopped = False
        self.delay_counter = 0
        self.plot_update_counter = 0
        self.plot_needs_update = True
        
        self.precalculate_positions()
        self.generate_ball_paths()
        self.start_next_balls()
    
    def start_next_balls(self):
        """Start new balls with proper positioning"""
        while (len(self.balls_in_flight) < self.max_simultaneous_balls and 
               self.balls_dropped < self.n_balls):
            ball_index = self.balls_dropped
            
            if (0, 0) in self.peg_positions:
                start_pos = self.peg_positions[(0, 0)]
                x, y = start_pos[0], self.board_area['y'] + 20
                target_x, target_y = start_pos
            else:
                x = self.board_area['x'] + self.board_area['width'] // 2
                y = self.board_area['y'] + 20
                target_x, target_y = x, self.board_area['y'] + self.board_margin
            
            x, y = self.clamp_position(x, y)
            target_x, target_y = self.clamp_position(target_x, target_y)
            
            self.balls_in_flight.append([ball_index, 0, x, y, target_x, target_y])
            self.balls_dropped += 1
    
    def get_peg_position(self, row, col):
        """Get pre-calculated peg position"""
        return self.peg_positions.get((row, col), (self.board_area['x'] + self.board_area['width']//2, self.board_area['y'] + self.board_margin))
    
    def get_bin_position(self, bin_num):
        """Get pre-calculated bin position"""
        if 0 <= bin_num < len(self.bin_positions):
            return self.bin_positions.get(bin_num, (self.board_area['x'] + self.board_area['width']//2, self.board_area['y'] + self.board_area['height'] - 60))
        return (self.board_area['x'] + self.board_area['width']//2, self.board_area['y'] + self.board_area['height'] - 60)
    
    def clamp_position(self, x, y):
        """Ball position clamping with reduced safe zone"""
        area = self.board_area
        
        safety_margin = self.board_margin  # Now smaller (40)
        min_x = area['x'] + safety_margin
        max_x = area['x'] + area['width'] - safety_margin
        min_y = area['y'] + 20
        max_y = area['y'] + area['height'] - 20
        
        clamped_x = max(min_x, min(x, max_x))
        clamped_y = max(min_y, min(y, max_y))
        
        if (clamped_x <= area['x'] + 10 or clamped_x >= area['x'] + area['width'] - 10 or
            clamped_y <= area['y'] + 10 or clamped_y >= area['y'] + area['height'] - 10):
            clamped_x = area['x'] + area['width'] // 2
            clamped_y = area['y'] + area['height'] // 2
        
        return clamped_x, clamped_y
    
    def update_animation(self):
        """Animation update with proper containment"""
        if not self.ball_falling or self.paused or self.stopped:
            return
        
        if self.delay_counter > 0:
            self.delay_counter -= 1
        else:
            self.start_next_balls()
            self.delay_counter = max(1, 30 - (self.ball_speed * 0.3))
        
        balls_to_remove = []
        
        for i, ball_data in enumerate(self.balls_in_flight):
            ball_index, current_row, x, y, target_x, target_y = ball_data
            
            x, y = self.clamp_position(x, y)
            target_x, target_y = self.clamp_position(target_x, target_y)
            
            dx = target_x - x
            dy = target_y - y
            distance_sq = dx * dx + dy * dy
            
            effective_speed = min(self.ball_speed * 2, 30)
            
            if distance_sq < effective_speed * effective_speed:
                x, y = self.clamp_position(target_x, target_y)
                
                if current_row < self.n_rows:
                    current_row += 1
                    if current_row < self.n_rows:
                        target_col = self.ball_paths[ball_index][current_row]
                        target_x, target_y = self.get_peg_position(current_row, target_col)
                    else:
                        final_position = self.ball_paths[ball_index][-1]
                        target_x, target_y = self.get_bin_position(final_position)
                    
                    target_x, target_y = self.clamp_position(target_x, target_y)
                    
                    self.balls_in_flight[i] = [ball_index, current_row, x, y, target_x, target_y]
                else:
                    final_position = self.ball_paths[ball_index][-1]
                    if 0 <= final_position < len(self.bins_count):
                        self.positions.append(final_position)
                        self.bins_count[final_position] += 1
                        self.balls_completed += 1
                        
                        if self.balls_completed % 5 == 0:
                            self.observed_mean = np.mean(self.positions)
                            self.observed_std = np.std(self.positions) if len(self.positions) > 1 else 0
                        
                        self.plot_update_counter += 1
                        if self.plot_update_counter >= self.plot_update_interval:
                            self.plot_needs_update = True
                            self.plot_update_counter = 0
                    
                    balls_to_remove.append(i)
            else:
                if distance_sq > 0:
                    distance = math.sqrt(distance_sq)
                    new_x = x + effective_speed * dx / distance
                    new_y = y + effective_speed * dy / distance
                else:
                    new_x, new_y = x, y
                
                new_x, new_y = self.clamp_position(new_x, new_y)
                
                self.balls_in_flight[i] = [ball_index, current_row, new_x, new_y, target_x, target_y]
        
        for index in reversed(balls_to_remove):
            self.balls_in_flight.pop(index)
        
        if self.balls_dropped >= self.n_balls and len(self.balls_in_flight) == 0:
            self.ball_falling = False
            self.simulation_complete = True
            self.plot_needs_update = True
    
    def update_plot_surface(self):
        """Create compact plots"""
        if not self.plot_needs_update or len(self.positions) == 0:
            return
        
        plt.style.use('default')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.2, 6), dpi=75)
        
        # Histogram
        counts, bins, patches = ax1.hist(self.positions, bins=self.n_rows + 1,
                                       range=(0, self.n_rows),
                                       color='skyblue', edgecolor='navy', alpha=0.8)
        ax1.set_title(f'Distribution\n({len(self.positions):,}/{self.n_balls:,})', fontsize=9, fontweight='bold')
        ax1.set_xlabel('Position', fontsize=8)
        ax1.set_ylabel('Count', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=7)
        
        # Horizontal bar chart for statistics
        categories = ['Mean', 'Std Dev']
        observed = [self.observed_mean, self.observed_std]
        expected = [self.theoretical_mean, self.theoretical_std]
        
        y_pos = np.arange(len(categories))
        bar_height = 0.35
        
        bars1 = ax2.barh(y_pos - bar_height/2, observed, bar_height, 
                        label='Observed', color='lightcoral', alpha=0.8)
        bars2 = ax2.barh(y_pos + bar_height/2, expected, bar_height, 
                        label='Expected', color='lightgreen', alpha=0.8)
        
        for i, (obs, exp) in enumerate(zip(observed, expected)):
            ax2.text(max(obs, exp) + 0.1, i - bar_height/2, f'{obs:.2f}', 
                    va='center', fontsize=7)
            ax2.text(max(obs, exp) + 0.1, i + bar_height/2, f'{exp:.2f}', 
                    va='center', fontsize=7)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(categories, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel('Value', fontsize=8)
        ax2.set_title('Statistics', fontsize=9, fontweight='bold')
        ax2.legend(fontsize=7)
        ax2.grid(axis='x', alpha=0.3)
        ax2.tick_params(labelsize=7)
        
        # Bin counts
        ax3.bar(range(len(self.bins_count)), self.bins_count, 
               color='orange', alpha=0.7, edgecolor='darkorange')
        ax3.set_title(f'Bin Counts\n({len(self.balls_in_flight)} falling)', 
                     fontsize=9, fontweight='bold')
        ax3.set_xlabel('Bin', fontsize=8)
        ax3.set_ylabel('Count', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=7)
        
        plt.tight_layout(pad=1.0)
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        
        raw_argb = renderer.tostring_argb()
        width, height = canvas.get_width_height()
        
        arr = np.frombuffer(raw_argb, dtype=np.uint8).reshape((height, width, 4))
        rgba = arr[:, :, [1, 2, 3, 0]]
        rgb = rgba[:, :, :3]
        rgb_bytes = rgb.tobytes()
        
        plt.close(fig)
        
        self.plot_surface = pygame.image.fromstring(rgb_bytes, (width, height), 'RGB')
        self.plot_needs_update = False
    
    def draw_horizontal_progress_bar(self):
        """Draw progress bar"""
        area = self.progress_area
        
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, 
                        (area['x'], area['y']+92, area['width'], area['height']))
        pygame.draw.rect(self.screen, self.DARK_GRAY, 
                        (area['x'], area['y']+92, area['width'], area['height']), 2)
        
        progress = len(self.positions) / self.n_balls if self.n_balls > 0 else 0
        progress_width = int((area['width'] - 20) * progress)
        
        if progress_width > 0:
            pygame.draw.rect(self.screen, self.GREEN, 
                           (area['x'] + 10, area['y'] + 100, progress_width, area['height'] - 16))
        
        remaining_width = (area['width'] - 20) - progress_width
        if remaining_width > 0:
            pygame.draw.rect(self.screen, self.LIGHT_BLUE, 
                           (area['x'] + 10 + progress_width, area['y'] + 100, remaining_width, area['height'] - 16))
        
        progress_text = f"{progress*100:.1f}%"
        if len(self.balls_in_flight) > 0:
            progress_text += f" | {len(self.balls_in_flight)} falling"
        
        text_surface = self.small_font.render(progress_text, True, self.BLACK)
        text_rect = text_surface.get_rect(center=(area['x'] + area['width']//2, area['y']+92 + area['height']//2))
        self.screen.blit(text_surface, text_rect)
    
    def draw_galton_board(self):
        """Draw Galton Board with reduced safe zone"""
        area = self.board_area
        
        # Board container
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, 
                        (area['x'], area['y'], area['width'], area['height']))
        pygame.draw.rect(self.screen, self.DARK_BLUE, 
                        (area['x'], area['y'], area['width'], area['height']), 4)
        
        safe_rect = (area['x'] + self.board_margin, area['y'] + self.board_margin, 
                    area['width'] - 2*self.board_margin, area['height'] - 2*self.board_margin)
        pygame.draw.rect(self.screen, (200, 255, 200), safe_rect, 1)  # Thinner line, smaller area
        
        # Title
        title = self.font.render('Galton Board Simulation', True, self.DARK_BLUE)
        title_rect = title.get_rect(center=(area['x'] + area['width']//2, area['y'] + 20))
        self.screen.blit(title, title_rect)
        
        # Draw pegs
        for (row, col), (x, y) in self.peg_positions.items():
            pygame.draw.circle(self.screen, self.BLACK, (x, y), 5)
            pygame.draw.circle(self.screen, self.GRAY, (x, y), 3)
        
        # Draw bins
        usable_width = area['width'] - (2 * self.board_margin)
        bin_width = usable_width / (self.n_rows + 1) * 0.7
        max_count = max(self.bins_count) if len(self.bins_count) > 0 and max(self.bins_count) > 0 else 1
        
        for bin_num in range(min(self.n_rows + 1, len(self.bins_count))):
            x, y = self.get_bin_position(bin_num)
            
            count = self.bins_count[bin_num] if bin_num < len(self.bins_count) else 0
            
            if count > 0:
                intensity = count / max_count
                color_val = int(255 * (1 - intensity * 0.6))
                bin_color = (color_val, color_val, 255)
            else:
                bin_color = self.LIGHT_BLUE
            
            # Draw bin
            pygame.draw.rect(self.screen, bin_color, 
                           (x - bin_width/2, y, bin_width, 60))
            pygame.draw.rect(self.screen, self.BLACK, 
                           (x - bin_width/2, y, bin_width, 60), 2)
            
            if count > 0:
                count_text = self.small_font.render(str(count), True, self.BLACK)
                count_rect = count_text.get_rect(center=(x, y + 30))
                pygame.draw.rect(self.screen, self.WHITE, 
                               (count_rect.x-2, count_rect.y-1, count_rect.width+4, count_rect.height+2))
                self.screen.blit(count_text, count_rect)
            
            bin_text = self.small_font.render(str(bin_num), True, self.DARK_GRAY)
            bin_rect = bin_text.get_rect(center=(x, y + 70))
            self.screen.blit(bin_text, bin_rect)
        
        # Draw animated balls
        ball_colors = [self.RED, self.YELLOW, self.ORANGE, self.GREEN, self.BLUE, 
                      (255, 0, 255), (0, 255, 255), (128, 255, 0)]
        
        for ball_data in self.balls_in_flight:
            ball_index, current_row, x, y, target_x, target_y = ball_data
            color = ball_colors[ball_index % len(ball_colors)]
            
            x, y = self.clamp_position(x, y)
            
            pygame.draw.circle(self.screen, self.WHITE, (int(x), int(y)), 8)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), 6)
            pygame.draw.circle(self.screen, self.BLACK, (int(x), int(y)), 6, 1)
    
    def draw_control_panel(self):
        """Draw control panel"""
        panel = self.slider_panel
        
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, 
                        (panel['x'], panel['y'], panel['width'], panel['height']))
        pygame.draw.rect(self.screen, self.DARK_GRAY, 
                        (panel['x'], panel['y'], panel['width'], panel['height']), 2)
        
        for name, slider in self.sliders.items():
            self.draw_slider(name, slider)
        
        for name, button in self.buttons.items():
            self.draw_button(name, button)
    
    def draw_slider(self, name, slider_info):
        """Draw professional slider with CORRECTED endpoint handling"""
        x, y, width, height = slider_info['x'], slider_info['y'], slider_info['width'], slider_info['height']
        min_val, max_val, value = slider_info['min'], slider_info['max'], slider_info['value']
        label = slider_info['label']
        unit = slider_info.get('unit', '')
        
        label_text = self.label_font.render(label, True, self.DARK_BLUE)
        self.screen.blit(label_text, (x, y - 30))
        
        track_rect = pygame.Rect(x, y + height//2 - 3, width, 6)
        pygame.draw.rect(self.screen, self.SLIDER_TRACK, track_rect)
        pygame.draw.rect(self.screen, self.DARK_GRAY, track_rect, 1)
        
        progress = (value - min_val) / (max_val - min_val)
        fill_width = int(width * progress)
        if fill_width > 0:
            fill_rect = pygame.Rect(x, y + height//2 - 3, fill_width, 6)
            pygame.draw.rect(self.screen, self.LIGHT_BLUE, fill_rect)
        
        handle_x = x + progress * width
        handle_y = y + height//2
        handle_color = self.SLIDER_ACTIVE if self.dragging == name else self.SLIDER_HANDLE
        
        pygame.draw.circle(self.screen, self.DARK_GRAY, (int(handle_x)+1, handle_y+1), 12)
        pygame.draw.circle(self.screen, handle_color, (int(handle_x), handle_y), 12)
        pygame.draw.circle(self.screen, self.WHITE, (int(handle_x), handle_y), 8)
        pygame.draw.circle(self.screen, handle_color, (int(handle_x), handle_y), 6)
        
        if name == 'prob':
            value_text = f"{value:.2f}"
        else:
            value_text = f"{int(value)}{unit}"
        
        value_surface = self.font.render(value_text, True, self.DARK_BLUE)
        value_rect = pygame.Rect(x + width + 15, y - 5, 40, 25)
        pygame.draw.rect(self.screen, self.WHITE, value_rect)
        pygame.draw.rect(self.screen, self.DARK_GRAY, value_rect, 1)
        value_text_rect = value_surface.get_rect(center=value_rect.center)
        self.screen.blit(value_surface, value_text_rect)
        
        min_text = self.small_font.render(str(min_val), True, self.GRAY)
        max_text = self.small_font.render(str(max_val), True, self.GRAY)
        self.screen.blit(min_text, (x, y + height + 5))
        self.screen.blit(max_text, (x + width - 30, y + height + 5))
        
        for i in range(5):
            tick_x = x + (width * i / 4)
            pygame.draw.line(self.screen, self.GRAY, (tick_x, y + height//2 + 3), (tick_x, y + height//2 + 10), 1)
    
    def draw_button(self, name, button_info):
        """Draw professional button"""
        x, y, width, height = button_info['x'], button_info['y'], button_info['width'], button_info['height']
        text = button_info['text']
        
        if name == 'run':
            button_color = self.GRAY if (self.ball_falling and not self.paused) else self.GREEN
            text = 'RUNNING' if (self.ball_falling and not self.paused) else 'START'
        elif name == 'pause':
            button_color = self.ORANGE if self.paused else self.BLUE
            text = 'RESUME' if self.paused else 'PAUSE'
        elif name == 'stop':
            button_color = self.RED
        else:
            button_color = self.BLUE
        
        shadow_rect = pygame.Rect(x + 2, y + 2, width, height)
        pygame.draw.rect(self.screen, self.DARK_GRAY, shadow_rect, border_radius=5)
        
        button_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, button_color, button_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.BLACK, button_rect, 2, border_radius=5)
        
        text_surface = self.font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=(x + width//2, y + height//2))
        self.screen.blit(text_surface, text_rect)
    
    def handle_slider_input(self, mouse_pos):
        """Handle slider interaction with CORRECTED endpoint handling"""
        if self.ball_falling and not self.stopped:
            return None
        
        mx, my = mouse_pos
        
        for name, slider in self.sliders.items():
            x, y, width, height = slider['x'], slider['y'], slider['width'], slider['height']
            
            if x <= mx <= x + width and y - 15 <= my <= y + height + 15:
                # CORRECTED: Allow full range including endpoints
                ratio = max(0.0, min(1.0, (mx - x) / width))  # Ensure exact 0.0 and 1.0 are possible
                new_value = slider['min'] + ratio * (slider['max'] - slider['min'])
                
                if name in ['balls', 'rows', 'speed', 'multi']:
                    if name == 'balls':
                        new_value = int(new_value / slider['step']) * slider['step']
                    else:
                        new_value = int(round(new_value))  # Use round() to ensure proper endpoint handling
                    
                    # CORRECTED: Ensure exact max values are reachable
                    slider['value'] = max(slider['min'], min(slider['max'], new_value))
                    
                    if name == 'balls':
                        self.n_balls = slider['value']
                    elif name == 'rows':
                        self.n_rows = slider['value']
                        self.update_bins_array_size()
                        self.precalculate_positions()
                    elif name == 'speed':
                        self.ball_speed = slider['value']
                    elif name == 'multi':
                        self.max_simultaneous_balls = slider['value']
                elif name == 'prob':
                    # CORRECTED: Ensure probability can reach exact endpoints
                    slider['value'] = max(slider['min'], min(slider['max'], round(new_value, 2)))
                    self.probability = slider['value']
                
                self.calculate_theoretical_values()
                return name
        return None
    
    def handle_button_click(self, mouse_pos):
        """Handle button clicks"""
        mx, my = mouse_pos
        
        for name, button in self.buttons.items():
            x, y, width, height = button['x'], button['y'], button['width'], button['height']
            
            if x <= mx <= x + width and y <= my <= y + height:
                if name == 'run' and not self.ball_falling:
                    self.start_animation()
                elif name == 'pause' and self.ball_falling:
                    self.paused = not self.paused
                elif name == 'stop':
                    self.ball_falling = False
                    self.stopped = True
                elif name == 'reset':
                    self.reset_parameters()
                return name
        return None
    
    def reset_parameters(self):
        """Reset parameters"""
        self.n_balls = 500
        self.n_rows = 12
        self.probability = 0.5
        self.ball_speed = 6
        self.max_simultaneous_balls = 8
        
        defaults = {'balls': 500, 'rows': 12, 'prob': 0.5, 'speed': 6, 'multi': 8}
        for key, default in defaults.items():
            self.sliders[key]['value'] = default
        
        self.ball_falling = False
        self.simulation_complete = False
        self.balls_completed = 0
        self.positions = []
        
        self.update_bins_array_size()
        self.bins_count.fill(0)
        
        self.balls_in_flight = []
        self.plot_surface = None
        self.paused = False
        self.stopped = False
        self.plot_needs_update = True
        
        self.calculate_theoretical_values()
        self.precalculate_positions()
    
    def run(self):
        """Main application loop"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.reset_parameters()
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        if not self.ball_falling:
                            self.start_animation()
                    elif event.key == pygame.K_p:
                        if self.ball_falling:
                            self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.dragging = self.handle_slider_input(event.pos)
                        if not self.dragging:
                            self.handle_button_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = None
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.handle_slider_input(event.pos)
            
            self.update_animation()
            
            if self.plot_needs_update:
                self.update_plot_surface()
            
            self.screen.fill(self.WHITE)
            
            # Header
            header_rect = pygame.Rect(0, 0, self.width, self.header_height)
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, header_rect)
            pygame.draw.rect(self.screen, self.DARK_GRAY, header_rect, 1)
            
            title = self.title_font.render('Galton Board Simulation', True, self.DARK_BLUE)
            title_rect = title.get_rect(center=(self.width//2, self.header_height//2))
            self.screen.blit(title, title_rect)
            
            # Draw all components
            self.draw_control_panel()
            self.draw_galton_board()
            
            if self.plot_surface:
                self.screen.blit(self.plot_surface, (self.plot_area['x'], self.plot_area['y']))
            
            self.draw_horizontal_progress_bar()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

# Main execution
if __name__ == "__main__":  
    simulation = GaltonBoardSimulation()
    simulation.run()
