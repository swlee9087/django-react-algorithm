from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
import pandas as pd
from icecream import ic
import json
import googlemaps



@dataclass
class ValueObject(object):

    train: object
    test: object
    id: str
    label: str
    fname: object
    context: str
    url: str
    dframe: object

    @property
    def fname(self) -> object: return self._fname
    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def train(self) -> object: return self._train
    @train.setter
    def train(self, train): self._train = train

    @property
    def test(self) -> object: return self._test
    @test.setter
    def test(self, test): self._test = test

    @property
    def id(self) -> str: return self._id
    @id.setter
    def id(self, id): self._id = id

    @property
    def label(self) -> str: return self._label
    @label.setter
    def label(self, label): self._label = label

    @property
    def context(self) -> str: return self._context
    @context.setter
    def context(self, context): self._context = context

    @property
    def url(self) -> str: return self._url
    @url.setter
    def url(self, url): self._url = url

    @property
    def dframe(self) -> object: return self._dframe
    @dframe.setter
    def dframe(self, dframe): self._dframe = dframe

    def create_model(self):
        return pd.read_csv(self.fname)

    def model_info(self, model):
        ic(model.head(3))
        ic(model.tail(3))
        ic(model.info())
        ic(model.describe())


class ReaderBase(metaclass=ABCMeta):
    @abstractmethod
    def new_file(self):
        pass

    @abstractmethod
    def csv(self):
        pass

    @abstractmethod
    def xls(self):
        pass

    @abstractmethod
    def json(self):
        pass


class PrinterBase(metaclass=ABCMeta):
    @abstractmethod
    def dframe(self):
        pass


class Reader(ReaderBase):
    def new_file(self, file) -> str:
        return file.context + file.fname

    def csv(self, file) -> object:
        return pd.read_csv(f'{file}.csv', encoding='UTF-8', thousands=',')

    def csv_header(self, file, header) -> object:
        return pd.read_csv(f'{file}.csv', encoding='UTF-8', thousands=',', header=header)

    def xls(self, file, header, usecols) -> object:
        return pd.read_excel(f'{file}.xls', header=header, usecols=usecols)

    def json(self, file) -> object:
        return json.load(open(f'{file}.json', encoding='UTF-8'))

    def gmaps(self) -> object:
        return googlemaps.Client(key='')


class Printer(PrinterBase):
    def dframe(self, this):
        ic(this.head(3))
        ic(this.tail(3))
        ic(this.info())
        ic(this.isnull().sum())
        # ic(this.columns())
        # ic(this.describe())