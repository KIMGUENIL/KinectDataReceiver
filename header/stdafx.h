
// stdafx.h : ���� ��������� ���� ��������� �ʴ�
// ǥ�� �ý��� ���� ���� �� ������Ʈ ���� ���� ������ 
// ��� �ִ� ���� �����Դϴ�.

#pragma once

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN            // ���� ������ �ʴ� ������ Windows ������� �����մϴ�.
#endif

#pragma comment(linker, "/entry:WinMainCRTStartup /subsystem:console")

#define _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES 1

#include "targetver.h"

#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      // �Ϻ� CString �����ڴ� ��������� ����˴ϴ�.

// MFC�� ���� �κа� ���� ������ ��� �޽����� ���� ����⸦ �����մϴ�.
#define _AFX_ALL_WARNINGS

#include <afxwin.h>         // MFC �ٽ� �� ǥ�� ���� ����Դϴ�.
#include <afxext.h>         // MFC Ȯ���Դϴ�.

#include <new>
#include <windows.h>
#include <strsafe.h>
#include <string>
#include <afxdisp.h>        // MFC �ڵ�ȭ Ŭ�����Դϴ�.



#ifndef _AFX_NO_OLE_SUPPORT
#include <afxdtctl.h>           // Internet Explorer 4 ���� ��Ʈ�ѿ� ���� MFC �����Դϴ�.
#endif
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>             // Windows ���� ��Ʈ�ѿ� ���� MFC �����Դϴ�.
#endif // _AFX_NO_AFXCMN_SUPPORT

#include <afxcontrolbars.h>     // MFC�� ���� �� ��Ʈ�� ���� ����
#include <afxsock.h> // MFC socket lib


// Case-insensitive wide-character char traits
struct i_char_traits : public std::char_traits<wchar_t>
{
	static bool eq(wchar_t c1, wchar_t c2) { return towupper(c1) == towupper(c2); }
	static bool ne(wchar_t c1, wchar_t c2) { return towupper(c1) != towupper(c2); }
	static bool lt(wchar_t c1, wchar_t c2) { return towupper(c1) <  towupper(c2); }
	static int compare(const wchar_t* s1, const wchar_t* s2, size_t n)
	{
		while (0 != n--)
		{
			if (towupper(*s1) < towupper(*s2)) return -1;
			if (towupper(*s1) > towupper(*s2)) return 1;
			++s1;
			++s2;
		}

		return 0;
	}

	static const wchar_t* find(const wchar_t* s, size_t n, wchar_t a)
	{
		int remaining = static_cast<int>(n);
		while ((--remaining >= 0) && (towupper(*s) != towupper(a)))
		{
			++s;
		}
		return (remaining >= 0) ? s : NULL;
	}
};

typedef std::basic_string<wchar_t, i_char_traits> wistring;


#ifdef _UNICODE
#if defined _M_IX86
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='x86' publicKeyToken='6595b64144ccf1df' language='*'\"")
#elif defined _M_X64
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='amd64' publicKeyToken='6595b64144ccf1df' language='*'\"")
#else
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif
#endif


