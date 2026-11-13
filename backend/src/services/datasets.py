from fastapi import HTTPException
from sqlalchemy import select
from src.models import Dataset
from src.services.business import metadata, round_label, has_rounds
from src.utils.serialization import checksum
import math, time
def create(db, data):
    name=data.name.strip()
    if len(name)<2: raise HTTPException(422, '批次名称去除空格后至少需要两个字符')
    if db.scalar(select(Dataset).where(Dataset.name==name)):
        raise HTTPException(409, '该数据批次名已存在，请修改名称')
    multi=has_rounds(data.records)
    for index,row in enumerate(data.records):
        for c in metadata()['columns']:
            value=row.get(c['key'])
            if c.get('optional'):
                if value is None or value=='': continue
                if not round_label(value) and not isinstance(value,str): raise HTTPException(422, f'第 {index+1} 行 {c["label"]} 格式不支持')
                continue
            if value is None or value=='':
                if multi and c['key'] in ('before','after'): continue # 多次测时批次的缺测行由分析标记，不再整批拒绝
                raise HTTPException(422, f'第 {index+1} 行缺少字段 {c["label"]}')
            if c['type']=='number':
                try:
                    if isinstance(value,bool) or not math.isfinite(float(value)): raise ValueError()
                    row[c['key']]=float(value)
                except (TypeError, ValueError, OverflowError):
                    if multi and c['key'] in ('before','after'): continue # 非法测时保留原值，分析时按缺测处理
                    raise HTTPException(422, f'第 {index+1} 行 {c["label"]} 必须为有限数值')
    d=Dataset(name=name, source=data.source, records=data.records, checksum=checksum(data.records), created_at=time.time())
    db.add(d); db.flush()
    return d
