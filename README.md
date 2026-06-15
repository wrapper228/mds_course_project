# ML Course Project

1) Поднимаем образ с постгрей docker compose up -d postgres, для этого на компе должен работать Docker 

2) Ставим .venv и устанавливаем pip install -r requirements.txt. Дальше из корня репозитория запускаем пайплайны:

root> python .\stage2_recommender_model\pipeline.py   
root> python .\stage3_return_model\pipeline.py    
