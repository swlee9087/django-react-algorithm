package shop.cofin.api.api.board.domain;

import com.sun.istack.NotNull;
import lombok.Data;
import org.springframework.stereotype.Component;
import shop.cofin.api.api.user.domain.User;
import shop.cofin.api.api.item.domain.Item;

import javax.persistence.*;

@Entity @Data @Component @Table(name = "articles")
public class Article {
    @Id
    @Column(name = "article_id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long articleId;

    @Column(length=50) @NotNull private String title;
    @Column(length=50) @NotNull private String content;
    @Column(name = "written_date", length=20) @NotNull private String writtenDate;

    @ManyToOne(fetch = FetchType.LAZY)
//    @JoinColumn(name = "user_id", insertable = false, updatable = false)
    @JoinColumn(name = "user_id")
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
//    @JoinColumn(name = "item_id", insertable = false, updatable = false)
    @JoinColumn(name = "item_id")
    private Item item;






}
