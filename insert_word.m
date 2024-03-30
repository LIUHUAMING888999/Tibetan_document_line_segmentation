function insert_word(filename,H)
%����MATLAB����Word�ĵ�
% ԭժ��xiezhh��������̳�ϵ���ؽ��飬������΢�ĸĶ�������
 
filespec_user =filename;
 
%===����word���ù���========================================================
try
Word = actxGetRunningServer('Word.Application');
catch
Word = actxserver('Word.Application');
end
Word.Visible = 1; % ʹwordΪ�ɼ�����set(Word, 'Visible', 1);
%===��word�ļ������·����û���򴴽�һ���հ��ĵ���========================
if exist(filespec_user,'file');
Document = Word.Documents.Open(filespec_user);
else
Document = Word.Documents.Add;
Document.SaveAs2(filespec_user);
end
%===��ʽ����===============================================================
Content = Document.Content;
Selection = Word.Selection;
Paragraphformat = Selection.ParagraphFormat;
%===�ĵ���ҳ�߾�===========================================================
Document.PageSetup.TopMargin = 20;
Document.PageSetup.BottomMargin = 45;
Document.PageSetup.LeftMargin = 45;
Document.PageSetup.RightMargin = 20;
%==========================================================================
 
%===�ĵ���ɲ���============================================================
% �ĵ��ı��⼰��ʽ
headline = '����';
Content.Start = 0; % ��ʼ��Ϊ0������ʾÿ��д�븲��֮ǰ����
Content.Text = headline;
Content.Font.Size = 16; % �����С
Content.Font.Bold = 1; % ����Ӵ�
Content.Paragraphs.Alignment = 'wdAlignParagraphCenter'; % ����,wdAlignParagraphLeft/Center/Right
 
% �ĵ��Ĵ���ʱ��
Selection.Start = Content.end; % ��ʼ�ĵط�����һ���Ľ�β
Selection.TypeParagraph; % ����һ���µĿն���
% 
% for i =1: length(ls('reverse_similar_patch\'))-2
% I_On= imread(['reverse_similar_patch\', num2str(i) '.png']);

% H=figure();
% imhist(I_On);
print(H,'-dbitmap');%��ͼƬ�������а�
Selection=Word.Selection;   
Selection.Range.Paste;%�ڵ�ǰ����λ�ò���ͼƬ

 
  
 
Document.ActiveWindow.ActivePane.View.Type = 'wdPrintView';
Document.Save; % �����ĵ�
Word.Quit; % �ر��ĵ�