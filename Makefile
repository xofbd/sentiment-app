SHELL:=/bin/bash
VENV := venv
DATA := data
ACTIVATE_VENV := source $(VENV)/bin/activate
MODEL := app/model/my_model.dill.gz

.PHONY: all
all: clean deploy

$(VENV): requirements.txt
	rm -rf $@
	python3 -m venv $@
	$(ACTIVATE_VENV) && pip install -r $<
	touch $@

$(DATA):
	mkdir -p $@

$(DATA)/Sentiment-Analysis-Dataset.zip: | $(DATA)
	wget -nc -P $| http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
	touch $@

$(MODEL): $(DATA)/Sentiment-Analysis-Dataset.zip | $(VENV)
	$(ACTIVATE_VENV) && python app/model/model.py --compress $< $@

.PHONY: deploy
deploy: $(MODEL) | $(VENV)
	$(ACTIVATE_VENV) && flask run

.PHONY: clean
clean:
	rm -rf $(VENV) $(DATA)
	rm -f $(MODEL)
	find . | grep __pycache__ | xargs rm -rf
