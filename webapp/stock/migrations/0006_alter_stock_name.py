# Generated by Django 4.0.3 on 2022-06-03 03:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stock', '0005_auto_20200703_1000'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stock',
            name='name',
            field=models.CharField(choices=[('LG에너지솔루션', 'LG에너지솔루션'), ('SK스퀘어', 'SK스퀘어'), ('카카오페이', '카카오페이'), ('현대중공업', '현대중공업'), ('크래프톤', '크래프톤'), ('PI첨단소재', 'PI첨단소재'), ('카카오뱅크', '카카오뱅크'), ('SK아이이테크놀로지', 'SK아이이테크놀로지'), ('SK바이오사이언스', 'SK바이오사이언스'), ('DL이앤씨', 'DL이앤씨'), ('명신산업', '명신산업'), ('하이브', '하이브'), ('에스케이바이오팜', '에스케이바이오팜'), ('한화시스템', '한화시스템'), ('지누스', '지누스'), ('두산퓨얼셀', '두산퓨얼셀'), ('포스코케미칼', '포스코케미칼'), ('더블유게임즈', '더블유게임즈'), ('우리금융지주', '우리금융지주'), ('효성첨단소재', '효성첨단소재'), ('효성티앤씨', '효성티앤씨'), ('HDC현대산업개발', 'HDC현대산업개발'), ('셀트리온', '셀트리온'), ('쿠쿠홈시스', '쿠쿠홈시스'), ('SK케미칼', 'SK케미칼'), ('BGF리테일', 'BGF리테일'), ('카카오', '카카오'), ('오리온', '오리온'), ('넷마블', '넷마블'), ('HD현대', 'HD현대'), ('두산밥캣', '두산밥캣'), ('삼성바이오로직스', '삼성바이오로직스'), ('화승엔터프라이즈', '화승엔터프라이즈'), ('동서', '동서'), ('LIG넥스원', 'LIG넥스원'), ('삼성물산', '삼성물산'), ('씨에스윈드', '씨에스윈드'), ('삼성에스디에스', '삼성에스디에스'), ('만도', '만도'), ('쿠쿠홀딩스', '쿠쿠홀딩스'), ('코스맥스', '코스맥스'), ('종근당', '종근당'), ('현대로템', '현대로템'), ('한진칼', '한진칼'), ('한국콜마', '한국콜마'), ('한국타이어앤테크놀로지', '한국타이어앤테크놀로지'), ('GS리테일', 'GS리테일'), ('신세계인터내셔날', '신세계인터내셔날'), ('한국항공우주', '한국항공우주'), ('이마트', '이마트'), ('메리츠금융지주', '메리츠금융지주'), ('BNK금융지주', 'BNK금융지주'), ('일진머티리얼즈', '일진머티리얼즈'), ('현대위아', '현대위아'), ('휠라홀딩스', '휠라홀딩스'), ('현대홈쇼핑', '현대홈쇼핑'), ('한미약품', '한미약품'), ('삼성생명', '삼성생명'), ('한화생명', '한화생명'), ('코오롱인더', '코오롱인더'), ('한전기술', '한전기술'), ('GKL', 'GKL'), ('SK', 'SK'), ('하이트진로', '하이트진로'), ('키움증권', '키움증권'), ('영원무역', '영원무역'), ('한세실업', '한세실업'), ('NAVER', 'NAVER'), ('KB금융', 'KB금융'), ('풍산', '풍산'), ('LG이노텍', 'LG이노텍'), ('LG유플러스', 'LG유플러스'), ('아시아나항공', '아시아나항공'), ('한전KPS', '한전KPS'), ('CJ제일제당', 'CJ제일제당'), ('팬오션', '팬오션'), ('SK이노베이션', 'SK이노베이션'), ('삼성카드', '삼성카드'), ('후성', '후성'), ('아모레퍼시픽', '아모레퍼시픽'), ('롯데관광개발', '롯데관광개발'), ('롯데쇼핑', '롯데쇼핑'), ('현대글로비스', '현대글로비스'), ('하나금융지주', '하나금융지주'), ('금호타이어', '금호타이어'), ('CJ CGV', 'CJ CGV'), ('GS', 'GS'), ('LG디스플레이', 'LG디스플레이'), ('기업은행', '기업은행'), ('강원랜드', '강원랜드'), ('한국금융지주', '한국금융지주'), ('엔씨소프트', '엔씨소프트'), ('현대백화점', '현대백화점'), ('대웅제약', '대웅제약'), ('티케이지휴켐스', '티케이지휴켐스'), ('한샘', '한샘'), ('LG전자', 'LG전자'), ('SNT모티브', 'SNT모티브'), ('신한지주', '신한지주'), ('코웨이', '코웨이'), ('LG생활건강', 'LG생활건강'), ('LG화학', 'LG화학'), ('대우건설', '대우건설'), ('포스코인터내셔널', '포스코인터내셔널'), ('대우조선해양', '대우조선해양'), ('현대두산인프라코어', '현대두산인프라코어'), ('두산에너빌리티', '두산에너빌리티'), ('한국가스공사', '한국가스공사'), ('케이티앤지', '케이티앤지'), ('한국조선해양', '한국조선해양'), ('대한유화', '대한유화'), ('케이티', '케이티'), ('제일기획', '제일기획'), ('SKC', 'SKC'), ('SK하이닉스', 'SK하이닉스'), ('삼성엔지니어링', '삼성엔지니어링'), ('한온시스템', '한온시스템'), ('한섬', '한섬'), ('현대엘리베이터', '현대엘리베이터'), ('에스원', '에스원'), ('HMM', 'HMM'), ('오뚜기', '오뚜기'), ('엘에스일렉트릭', '엘에스일렉트릭'), ('동원시스템즈', '동원시스템즈'), ('삼성중공업', '삼성중공업'), ('메리츠증권', '메리츠증권'), ('롯데케미칼', '롯데케미칼'), ('호텔신라', '호텔신라'), ('고려아연', '고려아연'), ('신풍제약', '신풍제약'), ('한올바이오파마', '한올바이오파마'), ('SK텔레콤', 'SK텔레콤'), ('현대모비스', '현대모비스'), ('현대해상', '현대해상'), ('현대그린푸드', '현대그린푸드'), ('한국전력공사', '한국전력공사'), ('녹십자', '녹십자'), ('한솔케미칼', '한솔케미칼'), ('동원산업', '동원산업'), ('에스엘', '에스엘'), ('보령', '보령'), ('부광약품', '부광약품'), ('한미사이언스', '한미사이언스'), ('POSCO홀딩스', 'POSCO홀딩스'), ('동국제강', '동국제강'), ('삼성증권', '삼성증권'), ('금호석유화학', '금호석유화학'), ('세방전지', '세방전지'), ('S-Oil', 'S-Oil'), ('한화에어로스페이스', '한화에어로스페이스'), ('현대제철', '현대제철'), ('KG스틸', 'KG스틸'), ('아이에스동서', '아이에스동서'), ('신세계', '신세계'), ('OCI', 'OCI'), ('현대건설', '현대건설'), ('현대미포조선', '현대미포조선'), ('GS건설', 'GS건설'), ('삼성SDI', '삼성SDI'), ('삼성전기', '삼성전기'), ('녹십자홀딩스', '녹십자홀딩스'), ('LS', 'LS'), ('SK네트웍스', 'SK네트웍스'), ('농심', '농심'), ('SK디스커버리', 'SK디스커버리'), ('한화', '한화'), ('영풍', '영풍'), ('넥센타이어', '넥센타이어'), ('롯데정밀화학', '롯데정밀화학'), ('DL', 'DL'), ('LX인터내셔널', 'LX인터내셔널'), ('태광산업', '태광산업'), ('DB하이텍', 'DB하이텍'), ('NH투자증권', 'NH투자증권'), ('미래에셋증권', '미래에셋증권'), ('삼성화재해상보험', '삼성화재해상보험'), ('오리온홀딩스', '오리온홀딩스'), ('삼성전자', '삼성전자'), ('쌍용씨앤이', '쌍용씨앤이'), ('현대자동차', '현대자동차'), ('한화솔루션', '한화솔루션'), ('롯데지주', '롯데지주'), ('기아', '기아'), ('효성', '효성'), ('CJ', 'CJ'), ('두산', '두산'), ('DB손해보험', 'DB손해보험'), ('대웅', '대웅'), ('영진약품', '영진약품'), ('케이씨씨', '케이씨씨'), ('롯데칠성음료', '롯데칠성음료'), ('아모레퍼시픽그룹', '아모레퍼시픽그룹'), ('대상', '대상'), ('LG', 'LG'), ('대한전선', '대한전선'), ('삼양홀딩스', '삼양홀딩스'), ('한국앤컴퍼니', '한국앤컴퍼니'), ('대한항공', '대한항공'), ('유한양행', '유한양행'), ('CJ대한통운', 'CJ대한통운')], max_length=30),
        ),
    ]
