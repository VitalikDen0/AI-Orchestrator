"""Image generation manager for Stable Diffusion models and LoRA."""

from __future__ import annotations

import json
import logging
import os
import sys
import subprocess
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)


def install_diffusers_dependencies():
    """
    Проверяет и устанавливает необходимые зависимости для diffusers
    """
    required_packages = [
        "diffusers", 
        "transformers", 
        "accelerate", 
        "safetensors", 
        "peft",
        "omegaconf"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.info(f"📦 Устанавливаю недостающие пакеты: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info("✅ Зависимости установлены")
        except Exception as e:
            logger.error(f"❌ Ошибка установки зависимостей: {e}")


class ImageGenerator:
    """
    Класс для генерации изображений с использованием Stable Diffusion
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.current_pipeline = None
        self.logger = logger

    def generate_image(self, prompt: str, negative_prompt: str, params: dict) -> Optional[str]:
        """
        Генерация изображения через прямую интеграцию со Stable Diffusion
        """
        start_time = time.time()
        
        # Логируем полученные параметры
        self.logger.info(f"🔧 Получены параметры генерации: prompt='{prompt[:50]}...', negative_prompt='{negative_prompt}'")
        
        # Горячая перезагрузка конфигурации LoRA
        self.model_manager.get_lora_config(force_reload=True)
        
        # Получаем путь к модели через ModelManager
        model_path = self.model_manager.get_model_path()
        if not model_path:
            self.logger.error("❌ Не удалось найти Stable Diffusion модель")
            return None
        
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Модель не найдена: {model_path}")
            return None
        
        # Определяем тип модели
        model_type = self.model_manager.detect_model_type(model_path)
        self.logger.info(f"🔍 Определен тип модели: {model_type} для {os.path.basename(model_path)}")
        
        # Применяем LoRA триггеры к промпту
        enhanced_prompt = self.model_manager.apply_lora_triggers(prompt, model_type)
        
        # Параметры по умолчанию
        default_params = {
            "seed": -1,
            "steps": 30,
            "width": 1024,
            "height": 1024,
            "cfg": 7.0,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras"
        }
        
        # Обновляем параметры пользовательскими значениями
        gen_params = default_params.copy()
        gen_params.update(params)
        
        # Исправляем seed если он -1
        if gen_params["seed"] == -1:
            gen_params["seed"] = random.randint(0, 2**32 - 1)
            self.logger.info(f"🎲 Сгенерирован случайный seed: {gen_params['seed']}")
        
        # Корректировка размеров для SD 1.5
        model_name = os.path.basename(model_path).lower()
        is_sdxl = any(keyword in model_name for keyword in ['xl', 'sdxl', 'illustrious', 'pony'])
        
        if not params.get("width") and not params.get("height"):
            if not is_sdxl:
                gen_params["width"] = 512
                gen_params["height"] = 512
                self.logger.info("📐 Автоматически установил размеры для SD 1.5 модели: 512x512")
        
        self.logger.info(f"🔧 Параметры генерации: {gen_params}")
        
        try:
            # Устанавливаем необходимые зависимости
            install_diffusers_dependencies()
            
            # Импортируем необходимые библиотеки
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline  # type: ignore
            from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline  # type: ignore
            from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler  # type: ignore
            import torch
            
            self.logger.info(f"📦 Загружаю модель: {model_path}")
            
            # Загружаем соответствующий pipeline
            if is_sdxl:
                self.logger.info("🎯 Обнаружена SDXL модель, использую StableDiffusionXLPipeline")
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            else:
                self.logger.info("🎯 Обнаружена SD 1.5 модель, использую StableDiffusionPipeline")
                pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            
            # Перемещаем на GPU если доступен
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                self.logger.info("🚀 Модель перемещена на GPU")
            else:
                self.logger.warning("⚠️ GPU недоступен, использую CPU")
            
            # Загружаем активные LoRA
            active_loras = self.model_manager.get_active_loras(model_type)
            if active_loras:
                self._load_loras(pipe, active_loras, model_type)
            
            # Сохраняем pipeline
            self.current_pipeline = pipe
            
            # Настраиваем scheduler
            if gen_params["sampler_name"] == "dpmpp_2m":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                self.logger.info("⚙️ Использую DPMSolverMultistepScheduler")
            
            # Генерируем изображение
            self.logger.info(f"🎨 Генерирую изображение: {enhanced_prompt[:50]}...")

            result = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=gen_params["steps"],
                guidance_scale=gen_params["cfg"],
                width=gen_params["width"],
                height=gen_params["height"],
                generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(gen_params["seed"])
            )

            # Получаем изображение
            image = self._extract_image_from_result(result)
            if image is None:
                raise RuntimeError('Не удалось получить изображение из результата pipeline')

            # Сохраняем изображение
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images", "generated")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"ConsoleTest_{gen_params['seed']}.png"
            output_path = os.path.join(output_dir, filename)
            
            image.save(output_path)
            self.logger.info(f"💾 Изображение сохранено: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации изображения: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        finally:
            # Очистка памяти
            if self.current_pipeline:
                del self.current_pipeline
                self.current_pipeline = None
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def generate_video(self, prompt: str, negative_prompt: str, params: dict) -> Optional[str]:
        """Генерация видео через прямую интеграцию со Stable Diffusion"""
        start_time = time.time()
        
        # Параметры по умолчанию для видео
        default_params = {
            "seed": -1,
            "steps": 20,
            "width": 512,
            "height": 512,
            "cfg": 7.0,
            "num_frames": 24,
            "fps": 8,
            "key_frames": 4
        }
        
        # Обновляем параметры пользовательскими значениями
        gen_params = default_params.copy()
        gen_params.update(params)
        
        # Исправляем seed если он -1
        if gen_params["seed"] == -1:
            gen_params["seed"] = random.randint(0, 2**32 - 1)
            self.logger.info(f"🎲 Сгенерирован случайный seed: {gen_params['seed']}")
        
        self.logger.info(f"🔧 Параметры генерации видео: {gen_params}")
        
        try:
            # Устанавливаем необходимые зависимости
            install_diffusers_dependencies()
            
            # Импортируем необходимые библиотеки
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
            from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
            from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
            import torch
            import numpy as np
            
            # Получаем путь к модели
            model_path = self.model_manager.get_model_path()
            if not model_path or not os.path.exists(model_path):
                self.logger.error("❌ Модель не найдена")
                return None
            
            model_type = self.model_manager.detect_model_type(model_path)
            is_sdxl = (model_type == 'sdxl')
            
            # Загружаем соответствующий pipeline
            if is_sdxl:
                self.logger.info("🎯 Обнаружена SDXL модель, использую StableDiffusionXLPipeline")
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            else:
                self.logger.info("🎯 Обнаружена SD 1.5 модель, использую StableDiffusionPipeline")
                pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                self.logger.info("🚀 Модель перемещена на GPU")
            
            # Загружаем LoRA
            active_loras = self.model_manager.get_active_loras(model_type)
            if active_loras:
                self._load_loras(pipe, active_loras, model_type)
            
            self.current_pipeline = pipe
            
            # Настраиваем scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            
            # Параметры генерации
            generation_config = {
                "width": gen_params["width"],
                "height": gen_params["height"],
                "num_inference_steps": gen_params["steps"],
                "guidance_scale": gen_params["cfg"],
                "num_images_per_prompt": 1
            }
            
            self.logger.info(f"🎬 Генерирую {gen_params['num_frames']} кадров для видео...")
            
            frames = []
            key_frames = gen_params["key_frames"]
            
            # Создаем вариации промпта для ключевых кадров
            key_prompts = [
                prompt,
                self._add_dynamic_elements(prompt, 1, key_frames),
                self._add_dynamic_elements(prompt, 2, key_frames),
                self._add_dynamic_elements(prompt, 3, key_frames)
            ]
            
            # Генерируем ключевые кадры
            for i in range(key_frames):
                seed = gen_params["seed"] + i * 50  # Разные seed'ы
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
                
                with torch.no_grad():
                    result = pipe(
                        prompt=key_prompts[i % len(key_prompts)],
                        negative_prompt=negative_prompt,
                        generator=generator,
                        **generation_config
                    )
                
                frame_img = self._extract_image_from_result(result)
                if frame_img is None:
                    raise RuntimeError('Не удалось получить кадр из результата pipeline')

                frames.append(frame_img)
                self.logger.info(f"  ✅ Ключевой кадр {i+1} готов")
            
            # Создаем интерполированные кадры между ключевыми кадрами
            final_frames = []
            frames_per_segment = gen_params["num_frames"] // (key_frames - 1) if key_frames > 1 else gen_params["num_frames"]
            
            if key_frames > 1:
                for segment in range(key_frames - 1):
                    img1 = np.array(frames[segment])
                    img2 = np.array(frames[segment + 1])
                    
                    for i in range(frames_per_segment):
                        t = i / frames_per_segment
                        t_smooth = 3 * t * t - 2 * t * t * t
                        interpolated_array = img1 * (1 - t_smooth) + img2 * t_smooth
                        interpolated_image = Image.fromarray(interpolated_array.astype(np.uint8))
                        final_frames.append(interpolated_image)
            else:
                final_frames = frames
            
            # Добавляем последний кадр если нужно
            while len(final_frames) < gen_params["num_frames"]:
                final_frames.append(final_frames[-1])
            
            final_frames = final_frames[:gen_params["num_frames"]]
            
            # Сохраняем видео
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_videos")
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"Video_{gen_params['seed']}.mp4"
            output_path = os.path.join(output_dir, filename)
            
            try:
                import imageio
                imageio.mimsave(output_path, final_frames, fps=gen_params["fps"])
                self.logger.info(f"💾 Видео сохранено: {output_path}")
                return output_path
            except ImportError:
                self.logger.warning("⚠️ imageio не установлен, сохраняю как GIF")
                output_path_gif = output_path.replace(".mp4", ".gif")
                final_frames[0].save(
                    output_path_gif,
                    save_all=True,
                    append_images=final_frames[1:],
                    duration=1000/gen_params["fps"],
                    loop=0
                )
                self.logger.info(f"💾 GIF сохранен: {output_path_gif}")
                return output_path_gif
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации видео: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        finally:
            self._unload_current_pipeline()

    def _load_loras(self, pipe, active_loras, model_type):
        """Загрузка LoRA адаптеров"""
        self.logger.info(f"🎭 Найдено {len(active_loras)} активных LoRA для типа {model_type}")
        
        # Проверяем доступность PEFT
        peft_available = False
        try:
            import peft
            peft_available = True
        except ImportError:
            self.logger.warning("⚠️ PEFT не установлен, LoRA могут не работать")
        
        loaded_loras = []
        for lora in active_loras:
            try:
                lora_filename = lora.get('filename', '')
                lora_strength = lora.get('strength', 1.0)
                lora_path = os.path.join(self.model_manager.lora_dir, model_type, lora_filename)
                
                if not os.path.exists(lora_path):
                    continue
                    
                adapter_name = os.path.splitext(lora_filename)[0]
                
                if lora_filename.endswith('.safetensors'):
                    if not peft_available:
                        continue
                    try:
                        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                        loaded_loras.append((adapter_name, lora_strength))
                        self.logger.info(f"✅ Загружена LoRA: {lora_filename} ({lora_strength})")
                    except Exception as e:
                        self.logger.error(f"❌ Ошибка загрузки LoRA {lora_filename}: {e}")
                else:
                    try:
                        pipe.load_lora_weights(lora_path)
                        loaded_loras.append((lora_filename, lora_strength))
                        self.logger.info(f"✅ Загружена LoRA (legacy): {lora_filename}")
                    except Exception as e:
                        self.logger.error(f"❌ Ошибка загрузки legacy LoRA {lora_filename}: {e}")
                        
            except Exception as e:
                self.logger.error(f"❌ Ошибка обработки LoRA {lora.get('filename')}: {e}")

        # Применяем веса
        if loaded_loras and hasattr(pipe, 'set_adapters'):
            try:
                adapter_names = [name for name, _ in loaded_loras]
                adapter_weights = [weight for _, weight in loaded_loras]
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                self.logger.info(f"⚙️ Настроены веса адаптеров")
            except Exception as e:
                self.logger.warning(f"⚠️ Не удалось настроить веса адаптеров: {e}")

    def _extract_image_from_result(self, result) -> Optional[Image.Image]:
        """Извлекает PIL Image из результата pipeline"""
        try:
            imgs = getattr(result, 'images', None)
            if imgs:
                return imgs[0]
            elif isinstance(result, (tuple, list)) and len(result) > 0:
                return result[0]
        except Exception:
            pass
        return None

    def _unload_current_pipeline(self):
        """Выгружает текущий pipeline для экономии VRAM"""
        try:
            if hasattr(self, 'current_pipeline') and self.current_pipeline is not None:
                self.logger.info("🔄 Выгружаю pipeline для экономии VRAM...")
                
                # Перемещаем на CPU
                if hasattr(self.current_pipeline, 'to'):
                    self.current_pipeline.to('cpu')
                
                # Удаляем pipeline
                del self.current_pipeline
                self.current_pipeline = None
                
                # Принудительная очистка памяти GPU
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.logger.info("🧹 Очищен кэш CUDA")
                except Exception as e:
                    self.logger.warning(f"⚠️ Не удалось очистить CUDA кэш: {e}")
                
                self.logger.info("✅ Pipeline выгружен")
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка выгрузки pipeline: {e}")

    def _is_realesrgan_available(self) -> bool:
        """Проверяет доступность модели RealESRGAN"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "stable_diff", "RealESRGAN_x4.pth")
            return os.path.exists(model_path)
        except Exception:
            return False

    def upscale_image_realesrgan(self, image_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Увеличивает изображение в 4 раза с помощью RealESRGAN
        """
        try:
            self.logger.info(f"📈 Начинаю апскейл изображения: {os.path.basename(image_path)}")
            
            # Путь к модели RealESRGAN
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "stable_diff", "RealESRGAN_x4.pth")
            
            if not os.path.exists(model_path):
                self.logger.info(f"ℹ️ Модель RealESRGAN не найдена: {model_path}")
                return None
            
            # Проверяем исходное изображение
            if not os.path.exists(image_path):
                self.logger.error(f"❌ Исходное изображение не найдено: {image_path}")
                return None
            
            # Определяем выходной путь
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_dir = os.path.dirname(image_path)
                output_path = os.path.join(output_dir, f"{base_name}_upscaled_4x.png")
            
            # Устанавливаем Real-ESRGAN если нужно
            self._install_realesrgan_dependencies()
            
            # Импортируем библиотеки
            try:
                import cv2
                import torch
                import numpy as np
                from PIL import Image
                
                # Пытаемся импортировать RealESRGAN
                try:
                    from realesrgan import RealESRGANer
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                except ImportError:
                    self.logger.warning("⚠️ realesrgan пакет не найден, использую альтернативный метод")
                    return self._upscale_image_alternative(image_path, output_path)
                
                # Настраиваем модель
                model = RRDBNet(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    num_feat=64, 
                    num_block=23, 
                    num_grow_ch=32, 
                    scale=4
                )
                
                # Создаем upsampler
                upsampler = RealESRGANer(
                    scale=4,
                    model_path=model_path,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=torch.cuda.is_available()
                )
                
                # Загружаем изображение
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Не удалось загрузить изображение: {image_path}")
                
                self.logger.info(f"📐 Исходный размер: {img.shape[1]}x{img.shape[0]}")
                
                # Выполняем апскейл
                self.logger.info("🚀 Выполняю апскейл...")
                output, _ = upsampler.enhance(img, outscale=4)
                
                # Сохраняем результат
                cv2.imwrite(output_path, output)
                
                self.logger.info(f"📐 Результирующий размер: {output.shape[1]}x{output.shape[0]}")
                self.logger.info(f"💾 Апскейл сохранен: {output_path}")
                
                return output_path
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка в процессе апскейла: {e}")
                return self._upscale_image_alternative(image_path, output_path)
                
        except Exception as e:
            self.logger.error(f"❌ Общая ошибка апскейла: {e}")
            return None
    
    def _upscale_image_alternative(self, image_path: str, output_path: str) -> Optional[str]:
        """
        Альтернативный метод апскейла с помощью простого бикубического интерполирования
        """
        try:
            self.logger.info("🔄 Использую альтернативный метод апскейла...")
            
            from PIL import Image
            
            # Загружаем изображение
            with Image.open(image_path) as img:
                original_size = img.size
                new_size = (original_size[0] * 4, original_size[1] * 4)
                
                # Увеличиваем с помощью бикубической интерполяции
                upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Сохраняем результат
                upscaled.save(output_path, "PNG")
                
                self.logger.info(f"📐 Увеличено с {original_size} до {new_size}")
                self.logger.info(f"💾 Альтернативный апскейл сохранен: {output_path}")
                
                return output_path
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка альтернативного апскейла: {e}")
            return None
    
    def _add_dynamic_elements(self, prompt, frame_index, total_frames):
        """Добавляет динамические элементы к промпту в зависимости от номера кадра"""
        
        # Базовые динамические элементы для разных типов промптов
        dynamic_elements = {
            "pose": [
                "slight head turn", "head turning", "looking to the side", "looking up", "looking down",
                "slight body movement", "body turning", "arm movement", "hand gesture", "finger movement",
                "eye movement", "blinking", "mouth movement", "smile change", "expression change"
            ],
            "lighting": [
                "slight lighting change", "light shift", "shadow movement", "highlight change",
                "ambient light variation", "light intensity change", "color temperature shift"
            ],
            "camera": [
                "slight camera movement", "camera angle change", "zoom effect", "perspective shift",
                "depth change", "focus adjustment", "blur variation"
            ],
            "motion": [
                "motion blur", "movement lines", "wind effect", "hair movement", "clothing movement",
                "particle effects", "energy flow", "magical effects", "sparkle effects"
            ]
        }
        
        # Определяем тип промпта
        prompt_lower = prompt.lower()
        
        # Выбираем подходящие динамические элементы
        if any(word in prompt_lower for word in ["anime", "girl", "boy", "character", "person"]):
            # Для персонажей добавляем движения и выражения
            elements = dynamic_elements["pose"] + dynamic_elements["motion"]
        elif any(word in prompt_lower for word in ["landscape", "nature", "scenery", "background"]):
            # Для пейзажей добавляем изменения освещения и камеры
            elements = dynamic_elements["lighting"] + dynamic_elements["camera"]
        else:
            # Для остальных используем все элементы
            elements = dynamic_elements["pose"] + dynamic_elements["lighting"] + dynamic_elements["camera"] + dynamic_elements["motion"]
        
        # Выбираем элемент в зависимости от номера кадра
        if elements:
            # Равномерно распределяем элементы по кадрам
            element_index = int((frame_index / total_frames) * len(elements))
            selected_element = elements[element_index % len(elements)]
            
            # Добавляем элемент к промпту
            enhanced_prompt = f"{prompt}, {selected_element}"
            
            # Добавляем интенсивность изменения в зависимости от прогресса
            progress = frame_index / total_frames
            if progress > 0.5:
                enhanced_prompt += ", subtle animation"
            
            return enhanced_prompt
        
        return prompt

    def _install_realesrgan_dependencies(self):
        """Устанавливает зависимости для RealESRGAN"""
        try:
            # Проверяем установлен ли basicsr
            try:
                import basicsr
            except ImportError:
                self.logger.info("📦 Устанавливаю basicsr...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'basicsr'], 
                             check=True, capture_output=True)
            
            # Проверяем установлен ли realesrgan
            try:
                import realesrgan
            except ImportError:
                self.logger.info("📦 Устанавливаю realesrgan...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'realesrgan'], 
                             check=True, capture_output=True)
                             
        except Exception as e:
            self.logger.warning(f"⚠️ Не удалось установить зависимости RealESRGAN: {e}")
        

class ModelManager:
    """
    Класс для управления Stable Diffusion моделями и LoRA
    """
    
    def __init__(self, base_dir: str | None = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir = base_dir
        self.stable_diff_dir = os.path.join(base_dir, "stable_diff")
        self.checkpoints_dir = os.path.join(self.stable_diff_dir, "checkpoints")
        self.lora_dir = os.path.join(self.stable_diff_dir, "lora")
        self.lora_config_path = os.path.join(self.lora_dir, "lora_config.json")
        
        # Кэш для конфигурации LoRA
        self._lora_config_cache = {}
        self._lora_config_last_modified = 0
        
        # Создаем папки если их нет
        self._ensure_directories()
        
        # Инициализируем конфигурацию LoRA
        self._init_lora_config()
    
    def _ensure_directories(self):
        """Создает необходимые папки если их нет"""
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(os.path.join(self.lora_dir, "sd"), exist_ok=True)
        os.makedirs(os.path.join(self.lora_dir, "sdxl"), exist_ok=True)
    
    def _init_lora_config(self):
        """Инициализирует конфигурацию LoRA"""
        if not os.path.exists(self.lora_config_path):
            self._generate_lora_config()
        else:
            self._scan_and_update_lora_config()
    
    def _scan_lora_files(self) -> Dict[str, List[str]]:
        """Сканирует папки LoRA и возвращает найденные файлы"""
        lora_files = {"sd": [], "sdxl": []}
        
        for model_type in ["sd", "sdxl"]:
            lora_type_dir = os.path.join(self.lora_dir, model_type)
            if os.path.exists(lora_type_dir):
                for file in os.listdir(lora_type_dir):
                    if file.lower().endswith(('.safetensors', '.ckpt', '.pt')):
                        lora_files[model_type].append(file)
        
        return lora_files
    
    def _generate_lora_config(self):
        """Генерирует базовую конфигурацию LoRA"""
        lora_files = self._scan_lora_files()
        config = {"loras": {}}
        
        for model_type, files in lora_files.items():
            for filename in files:
                lora_name = os.path.splitext(filename)[0]
                config["loras"][f"{model_type}_{lora_name}"] = {
                    "filename": filename,
                    "model_type": model_type,
                    "enabled": True,
                    "strength": 1.0,
                    "triggers": [],
                    "description": f"Auto-generated config for {filename}"
                }
        
        with open(self.lora_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Создан конфигурационный файл LoRA: {len(config['loras'])} файлов")
    
    def _scan_and_update_lora_config(self):
        """Сканирует LoRA файлы и обновляет конфигурацию новыми с анализом метаданных"""
        lora_files = self._scan_lora_files()
        
        try:
            with open(self.lora_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except:
            config = {"loras": {}}
        
        if "loras" not in config:
            config["loras"] = {}
        
        # Добавляем новые LoRA файлы
        updated = False
        for model_type, files in lora_files.items():
            for filename in files:
                lora_name = os.path.splitext(filename)[0]
                lora_key = f"{model_type}_{lora_name}"
                
                if lora_key not in config["loras"]:
                    # Анализируем метаданные LoRA
                    lora_path = os.path.join(self.lora_dir, model_type, filename)
                    metadata = self.analyze_lora_metadata(lora_path)
                    
                    # Определяем тип модели из метаданных или используем папку
                    detected_model_type = metadata.get("model_type", model_type)
                    if detected_model_type != "unknown" and detected_model_type != model_type:
                        logger.warning(f"⚠️ LoRA {filename} в папке {model_type}/, но метаданные указывают на {detected_model_type}")
                        # Используем тип из метаданных как более точный
                        actual_model_type = detected_model_type
                        lora_key = f"{actual_model_type}_{lora_name}"
                    else:
                        actual_model_type = model_type
                    
                    # Создаем конфигурацию с метаданными
                    config["loras"][lora_key] = {
                        "filename": filename,
                        "model_type": actual_model_type,
                        "enabled": True,
                        "strength": metadata.get("preferred_weight", 1.0),
                        "triggers": metadata.get("triggers", [])[:3],  # Берем топ-3 триггера
                        "description": metadata.get("description", f"Auto-detected: {metadata.get('base_model', 'Unknown')} LoRA"),
                        "base_model": metadata.get("base_model", "Unknown"),
                        "resolution": metadata.get("resolution", "Unknown"),
                        "author": metadata.get("author", ""),
                        "metadata_analyzed": True
                    }
                    updated = True
                    
                    logger.info(f"📋 Создана конфигурация для {filename}")
                    logger.info(f"   🎯 Тип: {actual_model_type} ({metadata.get('base_model', 'Unknown')})")
                    if metadata.get("triggers"):
                        logger.info(f"   🔤 Триггеры: {', '.join(metadata['triggers'][:3])}")
        
        if updated:
            with open(self.lora_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Обновлен конфигурационный файл LoRA с анализом метаданных")
    
    def get_lora_config(self, force_reload: bool = False) -> Dict:
        """Получает конфигурацию LoRA с кэшированием"""
        try:
            # Проверяем время модификации файла
            if os.path.exists(self.lora_config_path):
                mtime = os.path.getmtime(self.lora_config_path)
                
                # Если файл изменился или принудительная перезагрузка
                if force_reload or mtime > self._lora_config_last_modified:
                    with open(self.lora_config_path, 'r', encoding='utf-8') as f:
                        self._lora_config_cache = json.load(f)
                    self._lora_config_last_modified = mtime
                    logger.info("🔄 Перезагружена конфигурация LoRA")
                
                return self._lora_config_cache
            else:
                return {"loras": {}}
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки конфигурации LoRA: {e}")
            return {"loras": {}}
    
    def analyze_lora_metadata(self, lora_path: str) -> Dict[str, Any]:
        """
        Анализирует метаданные LoRA файла для определения совместимости
        
        Args:
            lora_path: Путь к LoRA файлу
            
        Returns:
            Словарь с метаданными LoRA
        """
        try:
            from safetensors import safe_open
            
            # Результат анализа
            metadata = {
                "model_type": "unknown",
                "base_model": "unknown", 
                "resolution": "unknown",
                "triggers": [],
                "description": "",
                "author": "",
                "version": "",
                "activation_text": "",
                "preferred_weight": 1.0
            }
            
            # Анализируем расширение файла
            file_ext = os.path.splitext(lora_path)[1].lower()
            
            if file_ext == ".safetensors":
                # Читаем метаданные из safetensors
                with safe_open(lora_path, framework="pt") as f:
                    metadata_raw = f.metadata()
                    
                    if metadata_raw:
                        # Извлекаем информацию о базовой модели
                        if "ss_base_model_version" in metadata_raw:
                            base_version = metadata_raw["ss_base_model_version"]
                            if "xl" in base_version.lower():
                                metadata["model_type"] = "sdxl"
                                metadata["base_model"] = "SDXL"
                            else:
                                metadata["model_type"] = "sd"
                                metadata["base_model"] = "SD 1.5"
                        
                        # Разрешение обучения
                        if "ss_resolution" in metadata_raw:
                            metadata["resolution"] = metadata_raw["ss_resolution"]
                        elif "ss_bucket_info" in metadata_raw:
                            try:
                                bucket_info = json.loads(metadata_raw["ss_bucket_info"])
                                if "buckets" in bucket_info:
                                    resolutions = list(bucket_info["buckets"].keys())
                                    if resolutions:
                                        metadata["resolution"] = resolutions[0]
                            except:
                                pass
                        
                        # Извлекаем теги и триггеры
                        if "ss_tag_frequency" in metadata_raw:
                            try:
                                tag_freq = json.loads(metadata_raw["ss_tag_frequency"])
                                # Получаем самые частые теги как потенциальные триггеры
                                all_tags = {}
                                for dataset_tags in tag_freq.values():
                                    all_tags.update(dataset_tags)
                                
                                # Сортируем по частоте и берем топ-5
                                sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
                                metadata["triggers"] = [tag for tag, _ in sorted_tags[:5]]
                            except:
                                pass
                        
                        # Другие поля метаданных
                        metadata_mapping = {
                            "ss_dataset_dirs": "description",
                            "modelspec.architecture": "architecture",
                            "modelspec.implementation": "implementation",
                            "modelspec.title": "title"
                        }
                        
                        for key, target in metadata_mapping.items():
                            if key in metadata_raw:
                                metadata[target] = metadata_raw[key]
                        
                        # Пытаемся извлечь автора и описание из названия файла
                        filename = os.path.basename(lora_path)
                        if "_" in filename or "-" in filename:
                            parts = filename.replace("_", " ").replace("-", " ").split()
                            metadata["author"] = parts[0] if parts else ""
                        
                        logger.info(f"🔍 Проанализированы метаданные LoRA: {filename}")
                        logger.info(f"   📋 Базовая модель: {metadata['base_model']}")
                        logger.info(f"   📐 Разрешение: {metadata['resolution']}")
                        if metadata["triggers"]:
                            logger.info(f"   🎯 Найденные триггеры: {', '.join(metadata['triggers'][:3])}")
            
            elif file_ext in [".ckpt", ".pt"]:
                # Для старых форматов используем эвристический анализ
                filename = os.path.basename(lora_path).lower()
                
                # Определяем тип по имени файла
                if any(keyword in filename for keyword in ["sdxl", "xl", "illustrious", "pony"]):
                    metadata["model_type"] = "sdxl"
                    metadata["base_model"] = "SDXL"
                else:
                    metadata["model_type"] = "sd"
                    metadata["base_model"] = "SD 1.5"
                
                logger.info(f"🔍 Анализ LoRA по имени файла: {metadata['base_model']}")
            
            return metadata
            
        except ImportError:
            logger.warning("⚠️ safetensors не установлен, анализ метаданных недоступен")
            return {"model_type": "unknown", "error": "safetensors not available"}
        except Exception as e:
            logger.error(f"❌ Ошибка анализа метаданных LoRA {lora_path}: {e}")
            return {"model_type": "unknown", "error": str(e)}
    
    def get_model_path(self) -> str:
        """Получает путь к модели с приоритетом .env > stable_diff"""
        # Приоритет 1: переменная окружения
        env_path = os.getenv('STABLE_DIFFUSION_MODEL_PATH', '').strip()
        if env_path and os.path.exists(env_path):
            return env_path
        
        # Приоритет 2: папка stable_diff/checkpoints
        if os.path.exists(self.checkpoints_dir):
            for file in os.listdir(self.checkpoints_dir):
                if file.lower().endswith(('.safetensors', '.ckpt')):
                    model_path = os.path.join(self.checkpoints_dir, file)
                    logger.info(f"🔍 Автоопределена модель: {file}")
                    return model_path
        
        # Fallback: возвращаем путь из .env даже если файл не существует
        return env_path if env_path else ""
    
    def detect_model_type(self, model_path: str) -> str:
        """
        Определяет тип модели (sd/sdxl) по метаданным или имени файла
        
        Args:
            model_path: Путь к checkpoint файлу
            
        Returns:
            Тип модели: 'sd' или 'sdxl'
        """
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Файл модели не найден: {model_path}")
            return 'sd'  # По умолчанию SD 1.5
        
        file_ext = os.path.splitext(model_path)[1].lower()
        model_name = os.path.basename(model_path).lower()
        
        # Сначала пытаемся анализировать метаданные
        if file_ext == ".safetensors":
            try:
                metadata = self.analyze_checkpoint_metadata(model_path)
                detected_type = metadata.get("model_type", "unknown")
                
                if detected_type != "unknown":
                    logger.info(f"🔍 Тип модели определен по метаданным: {detected_type}")
                    return detected_type
                    
            except Exception as e:
                logger.warning(f"⚠️ Ошибка анализа метаданных checkpoint: {e}")
        
        # Резервный анализ по имени файла
        if any(keyword in model_name for keyword in ['sdxl', 'xl', 'illustrious', 'pony']):
            logger.info(f"🔍 Тип модели определен по имени файла: sdxl")
            return 'sdxl'
        else:
            logger.info(f"🔍 Тип модели определен по имени файла: sd")
            return 'sd'
    
    def analyze_checkpoint_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Анализирует метаданные checkpoint файла
        
        Args:
            checkpoint_path: Путь к checkpoint файлу
            
        Returns:
            Словарь с метаданными checkpoint
        """
        try:
            from safetensors import safe_open
            
            metadata = {
                "model_type": "unknown",
                "architecture": "unknown",
                "base_model": "unknown",
                "resolution": "unknown",
                "model_name": "",
                "author": "",
                "description": "",
                "version": ""
            }
            
            file_ext = os.path.splitext(checkpoint_path)[1].lower()
            
            if file_ext == ".safetensors":
                with safe_open(checkpoint_path, framework="pt") as f:
                    metadata_raw = f.metadata()
                    tensor_keys = list(f.keys())
                    
                    logger.info(f"🔍 Найдено {len(tensor_keys)} тензоров в checkpoint")
                    if metadata_raw:
                        logger.info(f"🔍 Найдено {len(metadata_raw)} записей метаданных")
                    
                    # Анализируем ключи тензоров для определения архитектуры
                    sdxl_indicators = [
                        "conditioner.embedders.1.model.transformer.resblocks",
                        "conditioner.embedders.0.transformer.text_model",
                        "first_stage_model.encoder.down.0.block.0.norm1.weight",
                        "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
                    ]
                    
                    sd_indicators = [
                        "cond_stage_model.transformer.text_model.encoder.layers",
                        "first_stage_model.encoder.down.0.block.0.norm1.weight",
                        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"
                    ]
                    
                    # Ищем характерные ключи для SDXL
                    sdxl_score = 0
                    sd_score = 0
                    
                    for key in tensor_keys[:100]:  # Проверяем первые 100 ключей
                        for indicator in sdxl_indicators:
                            if indicator in key:
                                sdxl_score += 1
                                break
                        
                        for indicator in sd_indicators:
                            if indicator in key and "conditioner.embedders.1" not in key:
                                sd_score += 1
                                break
                    
                    # Дополнительная проверка по размерам моделей
                    try:
                        # Проверяем размер текстового энкодера
                        text_encoder_keys = [k for k in tensor_keys if "text_model.embeddings.token_embedding.weight" in k]
                        if text_encoder_keys:
                            tensor = f.get_tensor(text_encoder_keys[0])
                            vocab_size = tensor.shape[0]
                            logger.info(f"🔍 Размер словаря текстового энкодера: {vocab_size}")
                            
                            if vocab_size > 50000:  # SDXL обычно имеет больший словарь
                                sdxl_score += 2
                            else:
                                sd_score += 2
                    except:
                        pass
                    
                    # Проверяем размеры UNet
                    try:
                        unet_keys = [k for k in tensor_keys if "model.diffusion_model.input_blocks.0.0.weight" in k]
                        if unet_keys:
                            tensor = f.get_tensor(unet_keys[0])
                            input_channels = tensor.shape[1]
                            logger.info(f"🔍 Входные каналы UNet: {input_channels}")
                            
                            if input_channels == 4:  # Стандартно для обеих архитектур
                                # Проверяем другие размеры
                                output_channels = tensor.shape[0]
                                if output_channels >= 320:
                                    sdxl_score += 1
                    except:
                        pass
                    
                    logger.info(f"🔍 Счет определения: SDXL={sdxl_score}, SD={sd_score}")
                    
                    # Определяем тип модели на основе счета
                    if sdxl_score > sd_score:
                        metadata["model_type"] = "sdxl"
                        metadata["architecture"] = "SDXL"
                        metadata["base_model"] = "SDXL"
                        metadata["resolution"] = "1024x1024"
                    elif sd_score > 0:
                        metadata["model_type"] = "sd"
                        metadata["architecture"] = "SD 1.5"
                        metadata["base_model"] = "SD 1.5"
                        metadata["resolution"] = "512x512"
                    
                    # Извлекаем метаданные из заголовка файла
                    if metadata_raw:
                        # Стандартные поля
                        standard_fields = {
                            "modelspec.title": "model_name",
                            "modelspec.description": "description", 
                            "modelspec.author": "author",
                            "modelspec.implementation": "implementation",
                            "modelspec.architecture": "architecture_info"
                        }
                        
                        for raw_key, meta_key in standard_fields.items():
                            if raw_key in metadata_raw:
                                metadata[meta_key] = metadata_raw[raw_key]
                        
                        # Ищем другие полезные поля
                        for key, value in metadata_raw.items():
                            if "title" in key.lower() and not metadata.get("model_name"):
                                metadata["model_name"] = value
                            elif "description" in key.lower() and not metadata.get("description"):
                                metadata["description"] = value
                            elif "author" in key.lower() and not metadata.get("author"):
                                metadata["author"] = value
                    
                    logger.info(f"🔍 Финальное определение типа: {metadata['model_type']}")
                    if metadata["model_type"] != "unknown":
                        logger.info(f"   📋 Архитектура: {metadata['architecture']}")
                        logger.info(f"   📐 Разрешение: {metadata['resolution']}")
                        
                        if metadata.get("model_name"):
                            logger.info(f"   📝 Название: {metadata['model_name']}")
                    
            return metadata
            
        except ImportError:
            logger.warning("⚠️ safetensors не установлен, анализ checkpoint метаданных недоступен")
            return {"model_type": "unknown", "error": "safetensors not available"}
        except Exception as e:
            logger.error(f"❌ Ошибка анализа метаданных checkpoint {checkpoint_path}: {e}")
            return {"model_type": "unknown", "error": str(e)}
    
    def get_active_loras(self, model_type: str) -> List[Dict]:
        """Получает список активных LoRA для указанного типа модели"""
        config = self.get_lora_config()
        active_loras = []
        
        for lora_key, lora_config in config.get("loras", {}).items():
            if (lora_config.get("enabled", False) and 
                lora_config.get("model_type") == model_type):
                active_loras.append(lora_config)
        
        return active_loras
    
    def apply_lora_triggers(self, prompt: str, model_type: str) -> str:
        """Добавляет триггер-слова LoRA к промпту"""
        active_loras = self.get_active_loras(model_type)
        triggers = []
        
        for lora in active_loras:
            lora_triggers = lora.get("triggers", [])
            if lora_triggers:
                triggers.extend(lora_triggers)
        
        if triggers:
            trigger_text = ", ".join(triggers)
            enhanced_prompt = f"{prompt}, {trigger_text}"
            logger.info(f"🎯 Добавлены LoRA триггеры: {trigger_text}")
            return enhanced_prompt
        
        return prompt
    
    def analyze_all_loras(self) -> Dict[str, Dict[str, Any]]:
        """
        Анализирует метаданные всех LoRA файлов в системе
        
        Returns:
            Словарь с результатами анализа всех LoRA
        """
        results = {}
        lora_files = self._scan_lora_files()
        
        logger.info("🔍 Запускаю анализ метаданных всех LoRA файлов...")
        
        for model_type, files in lora_files.items():
            for filename in files:
                lora_path = os.path.join(self.lora_dir, model_type, filename)
                lora_key = f"{model_type}_{os.path.splitext(filename)[0]}"
                
                logger.info(f"📋 Анализирую: {filename}")
                metadata = self.analyze_lora_metadata(lora_path)
                
                results[lora_key] = {
                    "filename": filename,
                    "path": lora_path,
                    "folder_type": model_type,
                    "detected_type": metadata.get("model_type", "unknown"),
                    "base_model": metadata.get("base_model", "Unknown"),
                    "resolution": metadata.get("resolution", "Unknown"),
                    "triggers": metadata.get("triggers", []),
                    "author": metadata.get("author", ""),
                    "description": metadata.get("description", ""),
                    "compatible": metadata.get("model_type", model_type) == model_type,
                    "analysis_success": "error" not in metadata
                }
                
                # Предупреждение о несоответствии
                if (metadata.get("model_type", "unknown") != "unknown" and 
                    metadata.get("model_type") != model_type):
                    logger.warning(f"⚠️ {filename}: в папке {model_type}/, но предназначен для {metadata.get('model_type')}")
        
        logger.info(f"✅ Анализ завершен: {len(results)} LoRA файлов")
        return results
    
    def update_lora_metadata(self, force_update: bool = False) -> bool:
        """
        Обновляет метаданные существующих LoRA в конфигурации
        
        Args:
            force_update: Принудительно обновить все LoRA (даже уже проанализированные)
            
        Returns:
            True если конфигурация была обновлена
        """
        try:
            config = self.get_lora_config(force_reload=True)
            if "loras" not in config:
                config["loras"] = {}
            
            updated = False
            
            for lora_key, lora_config in config["loras"].items():
                # Пропускаем уже проанализированные LoRA (если не force_update)
                if not force_update and lora_config.get("metadata_analyzed", False):
                    continue
                
                filename = lora_config.get("filename")
                model_type = lora_config.get("model_type", "sd")
                
                if not filename:
                    continue
                
                # Ищем файл в соответствующей папке
                lora_path = os.path.join(self.lora_dir, model_type, filename)
                
                if not os.path.exists(lora_path):
                    logger.warning(f"⚠️ LoRA файл не найден: {lora_path}")
                    continue
                
                logger.info(f"🔍 Обновляю метаданные для {filename}")
                
                # Анализируем метаданные
                metadata = self.analyze_lora_metadata(lora_path)
                
                # Определяем актуальный тип модели
                detected_type = metadata.get("model_type", model_type)
                if detected_type != "unknown" and detected_type != model_type:
                    logger.warning(f"⚠️ LoRA {filename} в папке {model_type}/, но метаданные указывают на {detected_type}")
                    actual_model_type = detected_type
                    
                    # Создаем новый ключ с правильным типом
                    new_lora_key = f"{actual_model_type}_{os.path.splitext(filename)[0]}"
                    if new_lora_key != lora_key:
                        logger.info(f"🔄 Перемещаю конфигурацию: {lora_key} -> {new_lora_key}")
                        # Копируем в новый ключ
                        config["loras"][new_lora_key] = lora_config.copy()
                        # Удаляем старый ключ
                        del config["loras"][lora_key]
                        lora_key = new_lora_key
                        lora_config = config["loras"][lora_key]
                else:
                    actual_model_type = model_type
                
                # Сохраняем пользовательские настройки
                user_enabled = lora_config.get("enabled", True)
                user_strength = lora_config.get("strength", 1.0)
                user_triggers = lora_config.get("triggers", [])
                
                # Обновляем конфигурацию с метаданными
                config["loras"][lora_key].update({
                    "model_type": actual_model_type,
                    "enabled": user_enabled,  # Сохраняем пользовательскую настройку
                    "strength": user_strength,  # Сохраняем пользовательскую силу
                    "triggers": user_triggers if user_triggers else metadata.get("triggers", [])[:3],
                    "description": metadata.get("description", f"Auto-detected: {metadata.get('base_model', 'Unknown')} LoRA"),
                    "base_model": metadata.get("base_model", "Unknown"),
                    "resolution": metadata.get("resolution", "Unknown"),
                    "author": metadata.get("author", ""),
                    "metadata_analyzed": True
                })
                
                updated = True
                
                logger.info(f"✅ Обновлены метаданные для {filename}")
                logger.info(f"   🎯 Тип: {actual_model_type} ({metadata.get('base_model', 'Unknown')})")
                if metadata.get("triggers") and not user_triggers:
                    logger.info(f"   🔤 Новые триггеры: {', '.join(metadata['triggers'][:3])}")
            
            if updated:
                with open(self.lora_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ Конфигурация LoRA обновлена с метаданными")
                return True
            else:
                logger.info(f"📋 Все LoRA уже имеют актуальные метаданные")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка обновления метаданных LoRA: {e}")
            return False



__all__ = ["ModelManager", "ImageGenerator"]
