package shop.cofin.api.api.user.domain;

import lombok.*;
import shop.cofin.api.api.board.domain.Article;

import java.io.Serializable;
import java.util.*;

@Getter @Setter @ToString
public class UserDTO implements Serializable{
    private static final long serialVersionUID = 1L;

    private long userId;
    private String username;
    private String password;
    private String name;
    private String email;
    private String regDate;
    private List<Article> articleList;

    @Builder
    UserDTO(long userId, String username, String password, String name,
            String email, String regDate ){
        this.userId = userId;
        this.username = username;
        this.password = password;
        this.name = name;
        this.email = email;
        this.regDate = regDate;
    }
    @Builder
    UserDTO(long userId, String username, String password, String name,
            String email, String regDate, List<Article> articleList ){
        this.userId = userId;
        this.username = username;
        this.password = password;
        this.name = name;
        this.email = email;
        this.regDate = regDate;
        this.articleList = articleList;
    }

}
